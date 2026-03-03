import asyncio
import os
import uuid
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from minio.error import S3Error
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import get_db
from ..core.deps import get_current_member
from ..core.config import settings
from ..models import Member, Photo, UploadSession, YoloResultCache, PoleTypeResultCache, InferenceResult, InferenceDetection
from ..schemas import PhotoPresignIn, PhotoPresignOut, UploadResultOut, PhotoOut, InferenceResultOut
from ..core.storage import minio_client, MINIO_BUCKET
from ..services.upload import register_uploaded_photo, schedule_inference
from ..utils.inference import (
    select_nearest_yolo_payload,
    compact_inference_result_json,
    select_nearest_detection,
    build_selected_yolo_payload,
)

router = APIRouter(tags=["photos"])


def _debug(msg: str) -> None:
    if settings.API_DEBUG_LOG:
        print(msg)


@router.post("/photos/presign", response_model=PhotoPresignOut)
async def presign_photo_upload(
    data: PhotoPresignIn,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    _debug(
        f"[DEBUG] /photos/presign user={current.id} filename={data.filename} "
        f"img=({data.img_x},{data.img_y}) geo=({data.geo_x},{data.geo_y}) rdid={data.rdid}"
    )
    if not data.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 비어 있습니다.")
    if data.geo_x is None or data.geo_y is None:
        raise HTTPException(status_code=400, detail="geo_x, geo_y 좌표는 필수입니다.")
    if data.img_x is None or data.img_y is None:
        raise HTTPException(status_code=400, detail="img_x, img_y 좌표는 필수입니다.")
    if not data.rdid:
        raise HTTPException(status_code=400, detail="rdid는 필수입니다.")

    ext = Path(data.filename).suffix.lower()
    object_key = f"{current.id}/{uuid.uuid4().hex}{ext}"
    expires_in = 600  # 10분
    content_type = data.content_type or "application/octet-stream"

    upload_session = UploadSession(
        member_id=current.id,
        object_key=object_key,
        original_filename=data.filename,
        content_type=content_type,
        status="issued",
        img_x=data.img_x,
        img_y=data.img_y,
        geo_x=data.geo_x,
        geo_y=data.geo_y,
        geo_point=f"SRID=32652;POINT({data.geo_x} {data.geo_y})",
        rdid=data.rdid,
    )
    db.add(upload_session)
    try:
        await db.commit()
        await db.refresh(upload_session)
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=500, detail="업로드 세션 생성에 실패했습니다.")

    try:
        upload_url = await run_in_threadpool(
            minio_client.presigned_put_object,
            MINIO_BUCKET,
            object_key,
            expires=timedelta(seconds=expires_in),
        )
    except Exception as e:
        try:
            upload_session.status = "failed"
            await db.commit()
        except Exception:
            await db.rollback()
        raise HTTPException(status_code=502, detail=f"presigned URL 생성 실패: {e}") from e

    asyncio.create_task(
        register_uploaded_photo(
            session_id=upload_session.id,
            member_id=current.id,
            object_key=object_key,
            original_filename=data.filename,
            content_type=content_type,
        )
    )

    return PhotoPresignOut(
        session_id=str(upload_session.id),
        upload_url=upload_url,
        object_key=object_key,
        bucket=MINIO_BUCKET,
        expires_in=expires_in,
    )


@router.post("/photos", response_model=UploadResultOut, status_code=201)
async def upload_photo(
    file: UploadFile = File(...),
    img_x: float = Form(...),
    img_y: float = Form(...),
    geo_x: float = Form(...),
    geo_y: float = Form(...),
    rdid: str = Form(...),
    current: Member = Depends(get_current_member),
    db: AsyncSession = Depends(get_db),
):
    _debug(
        f"[DEBUG] /photos user={current.id} filename={file.filename} content_type={file.content_type} "
        f"img=({img_x},{img_y}) geo=({geo_x},{geo_y}) rdid={rdid}"
    )
    if not rdid:
        raise HTTPException(status_code=400, detail="rdid는 필수입니다.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 비어 있습니다.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    ext = Path(file.filename).suffix.lower()
    object_key = f"{current.id}/{uuid.uuid4().hex}{ext}"
    content_type = file.content_type or "application/octet-stream"

    try:
        await run_in_threadpool(
            minio_client.put_object,
            MINIO_BUCKET,
            object_key,
            file.file,
            -1,
            10 * 1024 * 1024,
            content_type=content_type,
        )
    except S3Error as e:
        raise HTTPException(status_code=502, detail=f"파일 업로드 실패: {e.code}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail="파일 업로드 중 오류가 발생했습니다.") from e

    try:
        file_size = os.fstat(file.file.fileno()).st_size
    except Exception:
        file_size = None

    dup_filters = [
        Photo.member_id == current.id,
        Photo.geo_x == geo_x,
        Photo.geo_y == geo_y,
        Photo.original_filename == file.filename,
    ]
    if file_size is not None:
        dup_filters.append(Photo.size_bytes == file_size)

    dup = await db.execute(select(Photo).where(*dup_filters))
    existing_photo = dup.scalar_one_or_none()
    if existing_photo:
        try:
            await run_in_threadpool(minio_client.remove_object, MINIO_BUCKET, object_key)
        except Exception:
            pass
        job_q = await db.execute(
            select(InferenceResult).where(InferenceResult.photo_id == existing_photo.id).order_by(InferenceResult.created_at.desc())
        )
        latest_job = job_q.scalar_one_or_none()
        yolo_cache = None
        if latest_job:
            yolo_q = await db.execute(
                select(YoloResultCache)
                .where(YoloResultCache.photo_id == existing_photo.id)
                .order_by(YoloResultCache.created_at.desc())
            )
            yolo_cache = yolo_q.scalar_one_or_none()

        pole_q = await db.execute(
            select(PoleTypeResultCache).where(PoleTypeResultCache.photo_id == existing_photo.id).order_by(PoleTypeResultCache.created_at.desc())
        )
        pole_cache = pole_q.scalar_one_or_none()

        inference_payload = None
        if latest_job:
            det_rows = await db.execute(
                select(InferenceDetection).where(InferenceDetection.job_id == latest_job.id)
            )
            dets = [
                {
                    "box_xyxy": d.box_xyxy,
                    "mask": d.mask,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                }
                for d in det_rows.scalars().all()
            ]
        else:
            dets = []

        if dets:
            selected_det = select_nearest_detection(dets, img_x, img_y)
            if selected_det:
                base = yolo_cache.result_json if yolo_cache and isinstance(yolo_cache.result_json, dict) else {}
                selected_yolo = build_selected_yolo_payload(base, selected_det)
                merged = {"yolo": selected_yolo}
                if pole_cache:
                    merged["pole_type"] = {
                        "status": pole_cache.status,
                        "result_object_key": pole_cache.result_object_key,
                        "result_json": pole_cache.result_json,
                        "error_message": pole_cache.error_message,
                        "size_bytes": None,
                    }
                inference_payload = InferenceResultOut(
                    id=str(existing_photo.id),
                    photo_id=str(existing_photo.id),
                    status="done" if pole_cache else "processing",
                    result_object_key=yolo_cache.result_object_key if yolo_cache else None,
                    result_json=compact_inference_result_json(merged),
                    error_message=None,
                    created_at=existing_photo.created_at,
                    updated_at=existing_photo.created_at,
                    started_at=None,
                    finished_at=None,
                    size_bytes=yolo_cache.result_json.get("size_bytes") if isinstance(yolo_cache.result_json, dict) else None,
                )
            else:
                inference_payload = InferenceResultOut(
                    id=str(existing_photo.id),
                    photo_id=str(existing_photo.id),
                    status="done",
                    result_object_key=None,
                    result_json="None",
                    error_message="no facility at point",
                    created_at=existing_photo.created_at,
                    updated_at=existing_photo.created_at,
                    started_at=None,
                    finished_at=None,
                    size_bytes=None,
                )
        elif yolo_cache and isinstance(yolo_cache.result_json, dict):
            selected_yolo = select_nearest_yolo_payload(yolo_cache.result_json, img_x, img_y)
            if selected_yolo:
                merged = {"yolo": selected_yolo}
                if pole_cache:
                    merged["pole_type"] = {
                        "status": pole_cache.status,
                        "result_object_key": pole_cache.result_object_key,
                        "result_json": pole_cache.result_json,
                        "error_message": pole_cache.error_message,
                        "size_bytes": None,
                    }
                inference_payload = InferenceResultOut(
                    id=str(existing_photo.id),
                    photo_id=str(existing_photo.id),
                    status="done" if pole_cache else "processing",
                    result_object_key=yolo_cache.result_object_key,
                    result_json=compact_inference_result_json(merged),
                    error_message=None,
                    created_at=existing_photo.created_at,
                    updated_at=existing_photo.created_at,
                    started_at=None,
                    finished_at=None,
                    size_bytes=yolo_cache.result_json.get("size_bytes") if isinstance(yolo_cache.result_json, dict) else None,
                )
            else:
                inference_payload = InferenceResultOut(
                    id=str(existing_photo.id),
                    photo_id=str(existing_photo.id),
                    status="done",
                    result_object_key=None,
                    result_json="None",
                    error_message="no facility at point",
                    created_at=existing_photo.created_at,
                    updated_at=existing_photo.created_at,
                    started_at=None,
                    finished_at=None,
                    size_bytes=None,
                )
        return UploadResultOut(
            duplicate=True,
            photo=PhotoOut(
                id=str(existing_photo.id),
                object_key=existing_photo.object_key,
                bucket=MINIO_BUCKET,
                original_filename=existing_photo.original_filename,
                content_type=existing_photo.content_type,
                size_bytes=existing_photo.size_bytes,
                created_at=existing_photo.created_at,
                img_x=existing_photo.img_x,
                img_y=existing_photo.img_y,
                geo_x=existing_photo.geo_x,
                geo_y=existing_photo.geo_y,
                rdid=existing_photo.rdid,
            ),
            inference=inference_payload,
        )

    photo = Photo(
        member_id=current.id,
        object_key=object_key,
        original_filename=file.filename,
        content_type=content_type,
        size_bytes=None,
        img_x=img_x,
        img_y=img_y,
        geo_x=geo_x,
        geo_y=geo_y,
        geo_point=f"SRID=32652;POINT({geo_x} {geo_y})",
        rdid=rdid,
    )

    db.add(photo)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        try:
            await run_in_threadpool(minio_client.remove_object, MINIO_BUCKET, object_key)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="업로드 메타데이터 저장에 실패했습니다.")

    await db.refresh(photo)

    try:
        stat = await run_in_threadpool(minio_client.stat_object, MINIO_BUCKET, object_key)
        photo.size_bytes = stat.size
        await db.commit()
        await db.refresh(photo)
    except Exception:
        pass

    try:
        await schedule_inference(photo, db)
    except Exception:
        pass

    return UploadResultOut(
        duplicate=False,
        photo=PhotoOut(
            id=str(photo.id),
            object_key=photo.object_key,
            bucket=MINIO_BUCKET,
            original_filename=photo.original_filename,
            content_type=photo.content_type,
            size_bytes=photo.size_bytes,
            created_at=photo.created_at,
            img_x=photo.img_x,
            img_y=photo.img_y,
            geo_x=photo.geo_x,
            geo_y=photo.geo_y,
            rdid=photo.rdid,
        ),
        inference=None,
    )
