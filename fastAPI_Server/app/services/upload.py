from __future__ import annotations

import asyncio
import traceback
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from minio.error import S3Error
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..celery_apps import celery_app, sam3_celery_app, ocr_celery_app
from ..core.config import settings
from ..core.db import SessionLocal
from ..models import ErrorLog, InferenceResult, Photo, UploadSession
from ..core.storage import MINIO_BUCKET, minio_client


def normalize_dt(dt):
    if not dt:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def inference_prefix(member_id: uuid.UUID) -> str:
    return str(member_id).lower()


def _result_no_detections(result_json: dict | None) -> bool | None:
    if not isinstance(result_json, dict):
        return None
    yolo = result_json.get("yolo")
    if isinstance(yolo, dict):
        return yolo.get("no_detections")
    return result_json.get("no_detections")


def _sub_status(result_json: dict | None, key: str) -> str | None:
    if not isinstance(result_json, dict):
        return None
    payload = result_json.get(key)
    if isinstance(payload, dict):
        return payload.get("status")
    return None


def resolve_inference_status(result_json: dict | None, fallback: str | None = None) -> str | None:
    no_detections = _result_no_detections(result_json)
    if no_detections is True:
        return "done"

    child_statuses = []
    for key in ("pole_type", "ocr"):
        status = _sub_status(result_json, key)
        if status:
            child_statuses.append(status)

    if any(status == "failed" for status in child_statuses):
        return "failed"
    if child_statuses and all(status == "done" for status in child_statuses):
        return "done"
    if child_statuses and any(status in ("queued", "processing", "done") for status in child_statuses):
        return "processing"
    return fallback


async def log_error(
    path: str | None,
    method: str | None,
    status_code: int | None,
    message: str | None,
    stacktrace: str | None = None,
):
    async with SessionLocal() as session:
        try:
            session.add(
                ErrorLog(
                    path=path,
                    method=method,
                    status_code=status_code,
                    message=message,
                    stacktrace=stacktrace,
                )
            )
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"[WARN] error log 저장 실패: {e}")


async def update_upload_session_status(
    upload_session: UploadSession,
    db: AsyncSession,
) -> None:
    if not upload_session:
        return
    infer_job = await db.get(InferenceResult, upload_session.job_id) if upload_session.job_id else None
    infer_status = infer_job.status if infer_job else None
    inferred_status = resolve_inference_status(
        infer_job.result_json if infer_job else None,
        fallback=infer_status,
    )

    if inferred_status == "failed":
        upload_session.status = "failed"
    elif inferred_status == "done":
        upload_session.status = "done"
    elif inferred_status in ("processing", "queued"):
        upload_session.status = "processing"


async def schedule_inference(photo: Photo, db: AsyncSession) -> InferenceResult:
    result_bucket = settings.INFERENCE_BUCKET
    result_prefix = inference_prefix(photo.member_id)
    job = InferenceResult(photo_id=photo.id, status="queued", rdid=photo.rdid)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    payload = {
        "job_id": str(job.id),
        "photo_id": str(photo.id),
        "bucket": MINIO_BUCKET,
        "object_key": photo.object_key,
        "rdid": photo.rdid,
        "img_x": photo.img_x,
        "img_y": photo.img_y,
        "result_bucket": result_bucket,
        "result_prefix": result_prefix,
    }
    try:
        await run_in_threadpool(
            celery_app.send_task,
            "inference_worker.tasks.run_inference",
            args=[],
            kwargs=payload,
            queue="inference",
        )
    except Exception as e:
        job.status = "failed"
        job.error_message = f"enqueue failed: {e}"
        await db.commit()
        await log_error(
            path="enqueue_inference",
            method=None,
            status_code=None,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="추론 작업 생성에 실패했습니다.") from e

    return job


async def schedule_pole_type(photo: Photo, inference_job_id: uuid.UUID, db: AsyncSession) -> None:
    result_bucket = settings.INFERENCE_BUCKET
    result_prefix = inference_prefix(photo.member_id)
    payload = {
        "job_id": str(inference_job_id),
        "photo_id": str(photo.id),
        "bucket": MINIO_BUCKET,
        "object_key": photo.object_key,
        "rdid": photo.rdid,
        "result_bucket": result_bucket,
        "result_prefix": result_prefix,
    }
    try:
        job = await db.get(InferenceResult, inference_job_id)
        if job:
            merged = dict(job.result_json or {})
            merged["pole_type"] = {
                "status": "queued",
                "result_object_key": None,
                "result_json": None,
                "error_message": None,
                "size_bytes": None,
            }
            job.result_json = merged
            await db.commit()
        await run_in_threadpool(
            sam3_celery_app.send_task,
            "sam3_worker.tasks.run_sam3",
            args=[],
            kwargs=payload,
            queue="sam3",
        )
    except Exception as e:
        job = await db.get(InferenceResult, inference_job_id)
        if job:
            merged = dict(job.result_json or {})
            merged["pole_type"] = {
                "status": "failed",
                "result_object_key": None,
                "result_json": None,
                "error_message": f"enqueue failed: {e}",
                "size_bytes": None,
            }
            job.result_json = merged
            job.status = resolve_inference_status(job.result_json, fallback="failed") or "failed"
            if job.status in ("done", "failed"):
                job.finished_at = datetime.now(timezone.utc)
            await db.commit()
        await log_error(
            path="enqueue_pole_type",
            method=None,
            status_code=None,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="pole_type 작업 생성에 실패했습니다.") from e


async def schedule_ocr(
    photo: Photo,
    inference_job_id: uuid.UUID,
    crop_items: list[dict],
    db: AsyncSession,
) -> None:
    result_prefix = inference_prefix(photo.member_id)
    payload = {
        "job_id": str(inference_job_id),
        "photo_id": str(photo.id),
        "rdid": photo.rdid,
        "crop_items": crop_items,
        "result_prefix": result_prefix,
    }
    try:
        job = await db.get(InferenceResult, inference_job_id)
        if job:
            merged = dict(job.result_json or {})
            merged["ocr"] = {
                "status": "queued",
                "result_json": None,
                "error_message": None,
                "size_bytes": None,
            }
            job.result_json = merged
            await db.commit()

        await run_in_threadpool(
            ocr_celery_app.send_task,
            "ocr_worker.tasks.run_ocr",
            args=[],
            kwargs=payload,
            queue="ocr",
        )
    except Exception as e:
        job = await db.get(InferenceResult, inference_job_id)
        if job:
            merged = dict(job.result_json or {})
            merged["ocr"] = {
                "status": "failed",
                "result_json": None,
                "error_message": f"enqueue failed: {e}",
                "size_bytes": None,
            }
            job.result_json = merged
            job.status = resolve_inference_status(job.result_json, fallback="failed") or "failed"
            if job.status in ("done", "failed"):
                job.finished_at = datetime.now(timezone.utc)
            await db.commit()
        await log_error(
            path="enqueue_ocr",
            method=None,
            status_code=None,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="ocr 작업 생성에 실패했습니다.") from e


async def register_uploaded_photo(session_id, member_id, object_key, original_filename, content_type):
    """
    presigned 업로드 완료 후 MinIO에 객체가 생기면 DB에 기록한다.
    presign 발급 시 백그라운드 태스크로 호출된다.
    """
    async with SessionLocal() as session:
        print(
            f"[DEBUG] register_uploaded_photo start session_id={session_id} member_id={member_id} "
            f"object_key={object_key} filename={original_filename} content_type={content_type}"
        )
        async def _update_status(status: str):
            try:
                us = await session.execute(select(UploadSession).where(UploadSession.id == session_id))
                upload_session = us.scalar_one_or_none()
                if upload_session:
                    upload_session.status = status
                    await session.commit()
            except Exception as e:
                await session.rollback()
                print(f"[WARN] upload_session 상태 업데이트 실패(session_id={session_id}, status={status}): {e}")

        for attempt in range(5):
            try:
                stat = await run_in_threadpool(minio_client.stat_object, MINIO_BUCKET, object_key)
                photo = None
                us_for_xy = await session.execute(select(UploadSession).where(UploadSession.id == session_id))
                xy = us_for_xy.scalar_one_or_none()
                if xy:
                    print(
                        f"[DEBUG] register_uploaded_photo xy session_id={session_id} "
                        f"img=({xy.img_x},{xy.img_y}) geo=({xy.geo_x},{xy.geo_y}) rdid={xy.rdid}"
                    )

                existing = await session.execute(
                    select(Photo).where(
                        Photo.member_id == member_id,
                        Photo.geo_x == (xy.geo_x if xy else 0.0),
                        Photo.geo_y == (xy.geo_y if xy else 0.0),
                        Photo.original_filename == original_filename,
                        Photo.size_bytes == stat.size,
                    )
                )
                photo = existing.scalar_one_or_none()
                if photo:
                    await run_in_threadpool(minio_client.remove_object, MINIO_BUCKET, object_key)
                else:
                    existing = await session.execute(select(Photo).where(Photo.object_key == object_key))
                    photo = existing.scalar_one_or_none()
                if not photo:
                    photo = Photo(
                        member_id=member_id,
                        object_key=object_key,
                        original_filename=original_filename,
                        content_type=content_type,
                        img_x=xy.img_x if xy else 0.0,
                        img_y=xy.img_y if xy else 0.0,
                        geo_x=xy.geo_x if xy else 0.0,
                        geo_y=xy.geo_y if xy else 0.0,
                        geo_point=f"SRID=32652;POINT({xy.geo_x if xy else 0.0} {xy.geo_y if xy else 0.0})",
                        rdid=xy.rdid if xy else None,
                        size_bytes=stat.size,
                        created_at=normalize_dt(stat.last_modified),
                    )
                    session.add(photo)
                    await session.flush()

                us = await session.execute(select(UploadSession).where(UploadSession.id == session_id))
                upload_session = us.scalar_one_or_none()
                existing_job = None
                if upload_session:
                    upload_session.status = "processing"
                    upload_session.photo_id = photo.id
                    upload_session.uploaded_at = normalize_dt(stat.last_modified)
                    if upload_session.status == "processing" and photo.id:
                        q = await session.execute(
                            select(InferenceResult)
                            .where(InferenceResult.photo_id == photo.id)
                            .order_by(InferenceResult.created_at.desc())
                        )
                        existing_job = q.scalar_one_or_none()
                        if existing_job:
                            upload_session.job_id = existing_job.id

                await session.commit()
                try:
                    if not existing_job:
                        job = await schedule_inference(photo, session)
                        if upload_session:
                            upload_session.job_id = job.id
                            upload_session.status = "queued"
                            await session.commit()
                except Exception:
                    await session.rollback()
                    await log_error(
                        path="enqueue_inference_or_pole_type",
                        method=None,
                        status_code=None,
                        message="failed to enqueue inference",
                        stacktrace=traceback.format_exc(),
                    )
                return
            except S3Error as e:
                if e.code == "NoSuchKey":
                    await asyncio.sleep(1)
                    continue
                print(f"[WARN] MinIO stat_object 실패(object_key={object_key}): {e}")
                await _update_status("failed")
                await log_error(
                    path="background:_register_uploaded_photo",
                    method=None,
                    status_code=None,
                    message=f"S3Error: {e}",
                    stacktrace=None,
                )
                return
            except Exception as e:
                await session.rollback()
                print(f"[WARN] presign 업로드 DB 기록 실패(object_key={object_key}): {e}")
                await _update_status("failed")
                await log_error(
                    path="background:_register_uploaded_photo",
                    method=None,
                    status_code=None,
                    message=str(e),
                    stacktrace=traceback.format_exc(),
                )
                return
        print(f"[WARN] presign 업로드 확인 실패(object_key={object_key})")
        await _update_status("missing")
        await log_error(
            path="background:_register_uploaded_photo",
            method=None,
            status_code=None,
            message="presigned upload not found in MinIO",
            stacktrace=None,
        )
