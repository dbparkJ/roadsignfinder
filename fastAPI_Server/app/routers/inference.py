import asyncio
import uuid
import traceback
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.db import get_db, SessionLocal
from ..models import InferenceResult, Photo, UploadSession, YoloResultCache, InferenceDetection
from ..schemas import InferenceResultOut, InferenceCallbackIn
from ..services.upload import (
    log_error,
    update_upload_sessions_status_by_job,
    schedule_pole_type,
    schedule_ocr,
    resolve_inference_status,
)
from ..utils.inference import (
    select_nearest_result_json,
    compact_inference_result_json,
    select_nearest_detection,
    build_selected_yolo_payload,
    mask_bbox,
)
from ..ocr_policy import build_ocr_queue_items

router = APIRouter(tags=["inference"])


def _debug(msg: str) -> None:
    if settings.API_DEBUG_LOG:
        print(msg)


@router.post("/inference/callback", status_code=204)
async def inference_callback(data: InferenceCallbackIn, request: Request):
    token = request.headers.get("x-inference-token")
    if token != settings.INFERENCE_CALLBACK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid callback token")

    job_id = uuid.UUID(data.job_id)
    now = datetime.now(timezone.utc)
    enqueue_pole_type = False
    enqueue_ocr = False
    enqueue_job_id = None
    enqueue_photo_id = None
    enqueue_crop_items: list[dict] = []
    try:
        async with SessionLocal() as db:
            async with db.begin():
                job = await db.get(InferenceResult, job_id, with_for_update=True)
                if not job:
                    raise HTTPException(status_code=404, detail="job not found")

                no_detections = None
                if isinstance(data.result_json, dict):
                    no_detections = data.result_json.get("no_detections")

                job.result_object_key = data.result_object_key
                if data.status == "done" and isinstance(data.result_json, dict):
                    if no_detections is False:
                        merged = dict(job.result_json or {})
                        merged["yolo"] = data.result_json
                        job.result_json = merged
                    else:
                        job.result_json = data.result_json
                else:
                    job.result_json = data.result_json

                if data.status == "done":
                    if no_detections is False:
                        job.status = resolve_inference_status(job.result_json, fallback="processing") or "processing"
                    else:
                        job.status = "done"
                else:
                    job.status = data.status
                job.error_message = data.error_message
                job.size_bytes = data.size_bytes
                if data.status == "processing" and not job.started_at:
                    job.started_at = now
                if job.status in ("failed", "done"):
                    job.finished_at = now
                job.updated_at = now
                if data.status == "done" and isinstance(data.result_json, dict):
                    db.add(
                        YoloResultCache(
                            photo_id=job.photo_id,
                            result_object_key=data.result_object_key,
                            result_json=data.result_json,
                        )
                    )
                    await db.execute(
                        delete(InferenceDetection).where(InferenceDetection.job_id == job.id)
                    )
                    boxes = data.result_json.get("boxes") or []
                    masks = data.result_json.get("masks") or []
                    _debug(
                        f"[DEBUG] inference_callback save_detections job_id={job.id} "
                        f"boxes={len(boxes)} masks={len(masks)}"
                    )
                    for idx, box in enumerate(boxes):
                        mask = masks[idx] if idx < len(masks) else None
                        xyxy = box.get("xyxy") if isinstance(box, dict) else None
                        if xyxy is None and mask is not None:
                            xyxy = list(mask_bbox(mask) or [])
                        db.add(
                            InferenceDetection(
                                job_id=job.id,
                                photo_id=job.photo_id,
                                box_xyxy=xyxy,
                                mask=mask,
                                class_id=box.get("class_id") if isinstance(box, dict) else None,
                                class_name=box.get("class_name") if isinstance(box, dict) else None,
                                confidence=box.get("confidence") if isinstance(box, dict) else None,
                            )
                    )

                if data.status == "done" and no_detections is False:
                    merged = dict(job.result_json or {})
                    enqueue_job_id = job.id
                    enqueue_photo_id = job.photo_id
                    if not isinstance(merged.get("pole_type"), dict):
                        enqueue_pole_type = True
                    crop_items = data.result_json.get("crop_images") if isinstance(data.result_json, dict) else None
                    masks = data.result_json.get("masks") if isinstance(data.result_json, dict) else None
                    if isinstance(crop_items, list) and crop_items and not isinstance(merged.get("ocr"), dict):
                        queue_items = build_ocr_queue_items(crop_items, masks)
                        target_items = [item for item in queue_items if item.get("ocr_target")]
                        if target_items:
                            enqueue_ocr = True
                            enqueue_crop_items = target_items
                        else:
                            merged["ocr"] = {
                                "status": "done",
                                "result_json": {
                                    "status": "skipped",
                                    "reason": "no_ocr_target_crops",
                                    "total": len(queue_items),
                                    "selected": 0,
                                },
                                "error_message": None,
                                "size_bytes": None,
                                "started_at": now.isoformat(),
                                "finished_at": now.isoformat(),
                                "updated_at": now.isoformat(),
                            }
                            job.result_json = merged
                            job.status = resolve_inference_status(job.result_json, fallback=job.status) or job.status
                            if job.status in ("done", "failed"):
                                job.finished_at = now
                    elif not isinstance(merged.get("ocr"), dict):
                        merged["ocr"] = {
                            "status": "done",
                            "result_json": {"status": "skipped", "reason": "no_crops"},
                            "error_message": None,
                            "size_bytes": None,
                            "started_at": now.isoformat(),
                            "finished_at": now.isoformat(),
                            "updated_at": now.isoformat(),
                        }
                        job.result_json = merged
                        job.status = resolve_inference_status(job.result_json, fallback=job.status) or job.status
                        if job.status in ("done", "failed"):
                            job.finished_at = now
                await update_upload_sessions_status_by_job(job.id, db)
        if enqueue_pole_type and enqueue_job_id and enqueue_photo_id:
            try:
                async with SessionLocal() as db2:
                    photo = await db2.get(Photo, enqueue_photo_id)
                    if photo:
                        await schedule_pole_type(photo, enqueue_job_id, db2)
            except Exception as e:
                await log_error(
                    path="inference_callback:schedule_pole_type",
                    method=request.method,
                    status_code=500,
                    message=str(e),
                    stacktrace=traceback.format_exc(),
                )
        if enqueue_ocr and enqueue_job_id and enqueue_photo_id:
            try:
                async with SessionLocal() as db3:
                    photo = await db3.get(Photo, enqueue_photo_id)
                    if photo:
                        await schedule_ocr(photo, enqueue_job_id, enqueue_crop_items, db3)
            except Exception as e:
                await log_error(
                    path="inference_callback:schedule_ocr",
                    method=request.method,
                    status_code=500,
                    message=str(e),
                    stacktrace=traceback.format_exc(),
                )
    except HTTPException:
        raise
    except Exception as e:
        await log_error(
            path="inference_callback",
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="callback handling failed") from e
    return


@router.get("/inference/{job_id}", response_model=InferenceResultOut)
async def get_inference_result(
    job_id: uuid.UUID,
    img_x: float | None = None,
    img_y: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    r = await db.execute(select(InferenceResult).where(InferenceResult.id == job_id))
    job = r.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    photo = await db.get(Photo, job.photo_id)
    sel_x = img_x if img_x is not None else (photo.img_x if photo else None)
    sel_y = img_y if img_y is not None else (photo.img_y if photo else None)
    selected_json = None
    if sel_x is not None and sel_y is not None:
        det_rows = await db.execute(
            select(InferenceDetection).where(InferenceDetection.job_id == job.id)
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
        _debug(
            f"[DEBUG] get_inference_result job_id={job.id} sel=({sel_x},{sel_y}) dets={len(dets)}"
        )
        selected_det = select_nearest_detection(dets, sel_x, sel_y)
        if selected_det:
            base = job.result_json.get("yolo") if isinstance(job.result_json, dict) else None
            if base is None and isinstance(job.result_json, dict):
                base = job.result_json
            yolo_selected = build_selected_yolo_payload(base if isinstance(base, dict) else {}, selected_det)
            if isinstance(job.result_json, dict) and "yolo" in job.result_json:
                selected_json = dict(job.result_json)
                selected_json["yolo"] = yolo_selected
            else:
                selected_json = yolo_selected
        else:
            selected_json = "None"
    if selected_json is None:
        selected_json = (
            select_nearest_result_json(job.result_json, sel_x, sel_y)
            if sel_x is not None and sel_y is not None
            else job.result_json
        )
    if selected_json == "None":
        return InferenceResultOut(
            id=str(job.id),
            photo_id=str(job.photo_id),
            status="done",
            result_object_key=None,
            result_json="None",
            error_message="no facility at point",
            rdid=photo.rdid if photo else job.rdid,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            size_bytes=None,
        )
    return InferenceResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=job.status,
        result_object_key=job.result_object_key,
        result_json=compact_inference_result_json(selected_json),
        error_message=job.error_message,
        rdid=job.rdid or (photo.rdid if photo else None),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        size_bytes=job.size_bytes,
    )


@router.get("/inference/result", response_model=InferenceResultOut)
async def get_inference_result_generic(
    job_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    photo_id: uuid.UUID | None = None,
    rdid: str | None = None,
    img_x: float | None = None,
    img_y: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    if not any([job_id, session_id, photo_id, rdid]):
        raise HTTPException(status_code=400, detail="job_id, session_id, photo_id, rdid 중 하나는 필요합니다.")

    job: InferenceResult | None = None

    upload_session = None
    if job_id:
        job = await db.get(InferenceResult, job_id)
    elif session_id:
        us = await db.execute(select(UploadSession).where(UploadSession.id == session_id))
        upload_session = us.scalar_one_or_none()
        if upload_session and upload_session.job_id:
            job = await db.get(InferenceResult, upload_session.job_id)
    elif photo_id:
        q = await db.execute(
            select(InferenceResult).where(InferenceResult.photo_id == photo_id).order_by(InferenceResult.created_at.desc())
        )
        job = q.scalars().first()
    elif rdid:
        q = await db.execute(
            select(InferenceResult).where(InferenceResult.rdid == rdid).order_by(InferenceResult.created_at.desc())
        )
        job = q.scalars().first()

    if not job:
        raise HTTPException(status_code=404, detail="inference result not found")

    photo = await db.get(Photo, job.photo_id)
    if img_x is not None and img_y is not None:
        sel_x, sel_y = img_x, img_y
    elif upload_session is not None:
        sel_x, sel_y = upload_session.img_x, upload_session.img_y
    else:
        sel_x = photo.img_x if photo else None
        sel_y = photo.img_y if photo else None
    selected_json = None
    if sel_x is not None and sel_y is not None:
        det_rows = await db.execute(
            select(InferenceDetection).where(InferenceDetection.job_id == job.id)
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
        _debug(
            f"[DEBUG] get_inference_result_generic job_id={job.id} sel=({sel_x},{sel_y}) dets={len(dets)}"
        )
        selected_det = select_nearest_detection(dets, sel_x, sel_y)
        if selected_det:
            base = job.result_json.get("yolo") if isinstance(job.result_json, dict) else None
            if base is None and isinstance(job.result_json, dict):
                base = job.result_json
            yolo_selected = build_selected_yolo_payload(base if isinstance(base, dict) else {}, selected_det)
            if isinstance(job.result_json, dict) and "yolo" in job.result_json:
                selected_json = dict(job.result_json)
                selected_json["yolo"] = yolo_selected
            else:
                selected_json = yolo_selected
        else:
            selected_json = "None"
    if selected_json is None:
        selected_json = (
            select_nearest_result_json(job.result_json, sel_x, sel_y)
            if sel_x is not None and sel_y is not None
            else job.result_json
        )
    if selected_json == "None":
        return InferenceResultOut(
            id=str(job.id),
            photo_id=str(job.photo_id),
            status="done",
            result_object_key=None,
            result_json="None",
            error_message="no facility at point",
            rdid=photo.rdid if photo else job.rdid,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            size_bytes=None,
        )

    return InferenceResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=job.status,
        result_object_key=job.result_object_key,
        result_json=compact_inference_result_json(selected_json),
        error_message=job.error_message,
        rdid=job.rdid or (photo.rdid if photo else None),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        size_bytes=job.size_bytes,
    )


@router.get("/uploads/{session_id}/inference", response_model=InferenceResultOut)
async def get_inference_by_session(
    session_id: uuid.UUID,
    img_x: float | None = None,
    img_y: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    us = await db.execute(select(UploadSession).where(UploadSession.id == session_id))
    upload_session = us.scalar_one_or_none()
    if not upload_session:
        raise HTTPException(status_code=404, detail="upload session not found")
    if not upload_session.job_id:
        return InferenceResultOut(
            id=str(upload_session.id),
            photo_id=str(upload_session.photo_id) if upload_session.photo_id else "",
            status=upload_session.status,
            result_object_key=None,
            result_json=None,
            error_message="job not created yet",
            created_at=upload_session.created_at,
            updated_at=upload_session.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    job = await db.get(InferenceResult, upload_session.job_id)
    if not job:
        return InferenceResultOut(
            id=str(upload_session.job_id),
            photo_id=str(upload_session.photo_id) if upload_session.photo_id else "",
            status="pending",
            result_object_key=None,
            result_json=None,
            error_message="job not found yet",
            created_at=upload_session.created_at,
            updated_at=upload_session.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    photo = await db.get(Photo, job.photo_id)
    if img_x is not None and img_y is not None:
        sel_x, sel_y = img_x, img_y
    else:
        sel_x, sel_y = upload_session.img_x, upload_session.img_y
    selected_json = None
    if sel_x is not None and sel_y is not None:
        det_rows = await db.execute(
            select(InferenceDetection).where(InferenceDetection.job_id == job.id)
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
        _debug(
            f"[DEBUG] get_inference_by_session job_id={job.id} sel=({sel_x},{sel_y}) dets={len(dets)}"
        )
        if not dets and job.status in ("queued", "processing"):
            return InferenceResultOut(
                id=str(job.id),
                photo_id=str(job.photo_id),
                status=job.status,
                result_object_key=None,
                result_json=None,
                error_message=None,
                rdid=photo.rdid if photo else job.rdid,
                created_at=job.created_at,
                updated_at=job.updated_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                size_bytes=job.size_bytes,
            )
        selected_det = select_nearest_detection(dets, sel_x, sel_y)
        if selected_det:
            base = job.result_json.get("yolo") if isinstance(job.result_json, dict) else None
            if base is None and isinstance(job.result_json, dict):
                base = job.result_json
            yolo_selected = build_selected_yolo_payload(base if isinstance(base, dict) else {}, selected_det)
            if isinstance(job.result_json, dict) and "yolo" in job.result_json:
                selected_json = dict(job.result_json)
                selected_json["yolo"] = yolo_selected
            else:
                selected_json = yolo_selected
        else:
            selected_json = "None"
    if selected_json is None:
        selected_json = (
            select_nearest_result_json(job.result_json, sel_x, sel_y)
            if sel_x is not None and sel_y is not None
            else job.result_json
        )
    if selected_json == "None":
        return InferenceResultOut(
            id=str(job.id),
            photo_id=str(job.photo_id),
            status="done",
            result_object_key=None,
            result_json="None",
            error_message="no facility at point",
            rdid=photo.rdid if photo else job.rdid,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            size_bytes=None,
        )

    return InferenceResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=job.status,
        result_object_key=job.result_object_key,
        result_json=compact_inference_result_json(selected_json),
        error_message=job.error_message,
        rdid=job.rdid or (photo.rdid if photo else None),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        size_bytes=job.size_bytes,
    )


@router.get("/inference/{job_id}/wait", response_model=InferenceResultOut)
async def wait_inference_result(
    job_id: uuid.UUID,
    timeout_seconds: int = 30,
    img_x: float | None = None,
    img_y: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
    while True:
        r = await db.execute(select(InferenceResult).where(InferenceResult.id == job_id))
        job = r.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status in ("done", "failed"):
            photo = await db.get(Photo, job.photo_id)
            sel_x = img_x if img_x is not None else (photo.img_x if photo else None)
            sel_y = img_y if img_y is not None else (photo.img_y if photo else None)
            selected_json = None
            if sel_x is not None and sel_y is not None:
                det_rows = await db.execute(
                    select(InferenceDetection).where(InferenceDetection.job_id == job.id)
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
                _debug(
                    f"[DEBUG] wait_inference_result job_id={job.id} sel=({sel_x},{sel_y}) dets={len(dets)}"
                )
                selected_det = select_nearest_detection(dets, sel_x, sel_y)
                if selected_det:
                    base = job.result_json.get("yolo") if isinstance(job.result_json, dict) else None
                    if base is None and isinstance(job.result_json, dict):
                        base = job.result_json
                    yolo_selected = build_selected_yolo_payload(base if isinstance(base, dict) else {}, selected_det)
                    if isinstance(job.result_json, dict) and "yolo" in job.result_json:
                        selected_json = dict(job.result_json)
                        selected_json["yolo"] = yolo_selected
                    else:
                        selected_json = yolo_selected
                else:
                    selected_json = "None"
            if selected_json is None:
                selected_json = (
                    select_nearest_result_json(job.result_json, sel_x, sel_y)
                    if sel_x is not None and sel_y is not None
                    else job.result_json
                )
            if selected_json == "None":
                return InferenceResultOut(
                    id=str(job.id),
                    photo_id=str(job.photo_id),
                    status="done",
                    result_object_key=None,
                    result_json="None",
                    error_message="no facility at point",
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    started_at=job.started_at,
                    finished_at=job.finished_at,
                )
            return InferenceResultOut(
                id=str(job.id),
                photo_id=str(job.photo_id),
                status=job.status,
                result_object_key=job.result_object_key,
                result_json=compact_inference_result_json(selected_json),
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        if datetime.now(timezone.utc) >= deadline:
            photo = await db.get(Photo, job.photo_id)
            sel_x = img_x if img_x is not None else (photo.img_x if photo else None)
            sel_y = img_y if img_y is not None else (photo.img_y if photo else None)
            selected_json = None
            if sel_x is not None and sel_y is not None:
                det_rows = await db.execute(
                    select(InferenceDetection).where(InferenceDetection.job_id == job.id)
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
                _debug(
                    f"[DEBUG] wait_inference_result timeout job_id={job.id} sel=({sel_x},{sel_y}) dets={len(dets)}"
                )
                selected_det = select_nearest_detection(dets, sel_x, sel_y)
                if selected_det:
                    base = job.result_json.get("yolo") if isinstance(job.result_json, dict) else None
                    if base is None and isinstance(job.result_json, dict):
                        base = job.result_json
                    yolo_selected = build_selected_yolo_payload(base if isinstance(base, dict) else {}, selected_det)
                    if isinstance(job.result_json, dict) and "yolo" in job.result_json:
                        selected_json = dict(job.result_json)
                        selected_json["yolo"] = yolo_selected
                    else:
                        selected_json = yolo_selected
                else:
                    selected_json = "None"
            if selected_json is None:
                selected_json = (
                    select_nearest_result_json(job.result_json, sel_x, sel_y)
                    if sel_x is not None and sel_y is not None
                    else job.result_json
                )
            if selected_json == "None":
                return InferenceResultOut(
                    id=str(job.id),
                    photo_id=str(job.photo_id),
                    status="done",
                    result_object_key=None,
                    result_json="None",
                    error_message="no facility at point",
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    started_at=job.started_at,
                    finished_at=job.finished_at,
                )
            return InferenceResultOut(
                id=str(job.id),
                photo_id=str(job.photo_id),
                status=job.status,
                result_object_key=job.result_object_key,
                result_json=compact_inference_result_json(selected_json),
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        await asyncio.sleep(1)
