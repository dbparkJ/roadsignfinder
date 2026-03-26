from __future__ import annotations

import traceback
import uuid
from datetime import timedelta
from pathlib import Path

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from minio.error import S3Error
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..celery_apps import sam3_correction_celery_app
from ..core.config import settings
from ..core.storage import MINIO_BUCKET, minio_client
from ..models import ClassCorrection, InferenceDetection, InferenceResult, Photo
from .upload import log_error


def _safe_token(value: str | None) -> str:
    if not value:
        return "unknown"
    token = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value).strip())
    token = token.strip("._")
    return token or "unknown"


def build_correction_image_object_key(
    correction_id: uuid.UUID | str,
    photo_name: str,
    rdid: str,
) -> str:
    ext = Path(photo_name).suffix.lower() or ".jpg"
    try:
        stem = uuid.UUID(str(correction_id)).hex
    except Exception:
        stem = _safe_token(str(correction_id))
    return f"{_safe_token(rdid)}/{stem}{ext}"


def build_correction_label_object_key(
    correction_id: uuid.UUID | str,
    photo_name: str,
    rdid: str,
) -> str:
    try:
        stem = uuid.UUID(str(correction_id)).hex
    except Exception:
        stem = _safe_token(str(correction_id))
    return f"{_safe_token(rdid)}/{stem}.txt"


async def ensure_correction_bucket() -> None:
    exists = await run_in_threadpool(minio_client.bucket_exists, settings.CLASS_CORRECTION_BUCKET)
    if not exists:
        await run_in_threadpool(minio_client.make_bucket, settings.CLASS_CORRECTION_BUCKET)


async def find_photo_by_rdid(
    db: AsyncSession,
    rdid: str,
    photo_name: str,
) -> Photo | None:
    exact = await db.execute(
        select(Photo)
        .where(Photo.rdid == rdid, Photo.original_filename == photo_name)
        .order_by(Photo.created_at.desc())
    )
    photo = exact.scalars().first()
    if photo:
        return photo

    fallback = await db.execute(
        select(Photo).where(Photo.rdid == rdid).order_by(Photo.created_at.desc())
    )
    return fallback.scalars().first()


async def latest_inference_for_photo(
    db: AsyncSession,
    photo_id: uuid.UUID,
) -> InferenceResult | None:
    result = await db.execute(
        select(InferenceResult)
        .where(InferenceResult.photo_id == photo_id)
        .order_by(InferenceResult.created_at.desc())
    )
    return result.scalars().first()


async def load_detection_payloads(
    db: AsyncSession,
    job_id: uuid.UUID,
) -> list[dict]:
    result = await db.execute(
        select(InferenceDetection)
        .where(InferenceDetection.job_id == job_id)
        .order_by(InferenceDetection.created_at.asc())
    )
    detections = []
    for idx, det in enumerate(result.scalars().all()):
        detections.append(
            {
                "index": idx,
                "id": str(det.id),
                "box_xyxy": det.box_xyxy,
                "mask": det.mask,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
            }
        )
    return detections


async def mark_correction_failed(
    correction: ClassCorrection,
    db: AsyncSession,
    *,
    path: str,
    method: str | None,
    message: str,
    stacktrace: str | None = None,
) -> None:
    correction.status = "failed"
    correction.error_message = message
    await db.commit()
    await log_error(
        path=path,
        method=method,
        status_code=500,
        message=message,
        stacktrace=stacktrace,
    )


async def prepare_unmatched_correction_upload(
    correction: ClassCorrection,
    content_type: str | None,
    db: AsyncSession,
    *,
    method: str | None,
) -> tuple[str, int]:
    object_key = build_correction_image_object_key(correction.id, correction.photo_name, correction.rdid)
    resolved_content_type = content_type or "application/octet-stream"
    expires_in = 600

    try:
        await ensure_correction_bucket()
        upload_url = await run_in_threadpool(
            minio_client.presigned_put_object,
            settings.CLASS_CORRECTION_BUCKET,
            object_key,
            expires=timedelta(seconds=expires_in),
        )
    except S3Error as e:
        await mark_correction_failed(
            correction,
            db,
            path="class_corrections:presign_unmatched",
            method=method,
            message=f"MinIO presign failed: {e.code}",
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail=f"correction image presign failed: {e.code}") from e
    except Exception as e:
        await mark_correction_failed(
            correction,
            db,
            path="class_corrections:presign_unmatched",
            method=method,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="correction image presign failed") from e

    correction.upload_bucket = settings.CLASS_CORRECTION_BUCKET
    correction.upload_image_object_key = object_key
    correction.status = "upload_required"
    correction.error_message = None
    if correction.result_json is None:
        correction.result_json = {}
    correction.result_json["content_type"] = resolved_content_type
    await db.commit()
    return upload_url, expires_in


async def confirm_unmatched_correction_upload(
    correction: ClassCorrection,
    db: AsyncSession,
    *,
    method: str | None,
) -> None:
    if not correction.upload_bucket or not correction.upload_image_object_key:
        raise HTTPException(status_code=400, detail="no pending upload target for this correction")

    try:
        await run_in_threadpool(
            minio_client.stat_object,
            correction.upload_bucket,
            correction.upload_image_object_key,
        )
    except S3Error as e:
        if e.code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="uploaded file not found in MinIO") from e
        await mark_correction_failed(
            correction,
            db,
            path="class_corrections:confirm_upload",
            method=method,
            message=f"MinIO stat failed: {e.code}",
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail=f"failed to verify uploaded file: {e.code}") from e
    except Exception as e:
        await mark_correction_failed(
            correction,
            db,
            path="class_corrections:confirm_upload",
            method=method,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="failed to verify uploaded file") from e

    correction.status = "uploaded"
    correction.error_message = None
    await db.commit()


async def schedule_correction_worker(
    correction: ClassCorrection,
    photo: Photo,
    inference_job: InferenceResult | None,
    detections: list[dict],
    db: AsyncSession,
) -> None:
    image_object_key = build_correction_image_object_key(correction.id, correction.photo_name, correction.rdid)
    label_object_key = build_correction_label_object_key(correction.id, correction.photo_name, correction.rdid)
    payload = {
        "correction_id": str(correction.id),
        "photo_id": str(photo.id),
        "source_bucket": MINIO_BUCKET,
        "source_object_key": photo.object_key,
        "photo_name": correction.photo_name,
        "rdid": correction.rdid,
        "class_name": correction.class_name,
        "img_x": correction.img_x,
        "img_y": correction.img_y,
        "existing_detections": detections,
        "upload_bucket": settings.CLASS_CORRECTION_BUCKET,
        "upload_image_object_key": image_object_key,
        "upload_label_object_key": label_object_key,
    }

    correction.photo_id = photo.id
    correction.inference_job_id = inference_job.id if inference_job else None
    correction.source_bucket = MINIO_BUCKET
    correction.source_object_key = photo.object_key
    correction.upload_bucket = settings.CLASS_CORRECTION_BUCKET
    correction.upload_image_object_key = image_object_key
    correction.upload_label_object_key = label_object_key
    correction.status = "queued"
    correction.error_message = None
    await db.commit()

    try:
        await run_in_threadpool(
            sam3_correction_celery_app.send_task,
            "sam3_correction_worker.tasks.run_correction",
            args=[],
            kwargs=payload,
            queue="sam3_correction",
        )
    except Exception as e:
        await mark_correction_failed(
            correction,
            db,
            path="class_corrections:schedule_worker",
            method=None,
            message=f"enqueue failed: {e}",
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="correction worker enqueue failed") from e
