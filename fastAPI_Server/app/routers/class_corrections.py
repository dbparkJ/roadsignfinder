import uuid
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.db import SessionLocal, get_db
from ..core.storage import MINIO_BUCKET
from ..models import ClassCorrection
from ..schemas import ClassCorrectionCallbackIn, ClassCorrectionCreateIn, ClassCorrectionOut
from ..services.class_corrections import (
    confirm_unmatched_correction_upload,
    find_photo_by_rdid,
    latest_inference_for_photo,
    load_detection_payloads,
    prepare_unmatched_correction_upload,
    schedule_correction_worker,
)
from ..services.upload import log_error

router = APIRouter(tags=["class_corrections"])


def _to_out(
    correction: ClassCorrection,
    *,
    upload_url: str | None = None,
    expires_in: int | None = None,
) -> ClassCorrectionOut:
    return ClassCorrectionOut(
        id=str(correction.id),
        photo_name=correction.photo_name,
        class_name=correction.class_name,
        rdid=correction.rdid,
        photo_id=str(correction.photo_id) if correction.photo_id else None,
        inference_job_id=str(correction.inference_job_id) if correction.inference_job_id else None,
        img_x=correction.img_x,
        img_y=correction.img_y,
        source_bucket=correction.source_bucket,
        source_object_key=correction.source_object_key,
        upload_bucket=correction.upload_bucket,
        upload_image_object_key=correction.upload_image_object_key,
        upload_label_object_key=correction.upload_label_object_key,
        status=correction.status,
        error_message=correction.error_message,
        result_json=correction.result_json,
        upload_url=upload_url,
        expires_in=expires_in,
        created_at=correction.created_at,
        updated_at=correction.updated_at,
        started_at=correction.started_at,
        finished_at=correction.finished_at,
    )


@router.post("/class-corrections", response_model=ClassCorrectionOut, status_code=201)
async def create_class_correction(
    data: ClassCorrectionCreateIn,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    photo_name = data.photo_name.strip()
    class_name = data.class_name.strip()
    rdid = data.rdid.strip()

    if not photo_name:
        raise HTTPException(status_code=400, detail="photo_name is required")
    if not class_name:
        raise HTTPException(status_code=400, detail="class_name is required")
    if not rdid:
        raise HTTPException(status_code=400, detail="rdid is required")

    correction = ClassCorrection(
        photo_name=photo_name,
        class_name=class_name,
        rdid=rdid,
        status="received",
    )
    db.add(correction)
    await db.commit()
    await db.refresh(correction)

    photo = await find_photo_by_rdid(db, rdid, photo_name)
    if photo:
        correction.img_x = photo.img_x
        correction.img_y = photo.img_y
        correction.photo_id = photo.id
        correction.source_bucket = MINIO_BUCKET
        correction.source_object_key = photo.object_key
        await db.commit()

        inference_job = await latest_inference_for_photo(db, photo.id)
        detections = await load_detection_payloads(db, inference_job.id) if inference_job else []
        await schedule_correction_worker(correction, photo, inference_job, detections, db)
        await db.refresh(correction)
        return _to_out(correction)

    upload_url, expires_in = await prepare_unmatched_correction_upload(
        correction,
        data.content_type,
        db,
        method=request.method,
    )
    await db.refresh(correction)
    return _to_out(correction, upload_url=upload_url, expires_in=expires_in)


@router.post("/class-corrections/{correction_id}/upload-complete", response_model=ClassCorrectionOut)
async def complete_class_correction_upload(
    correction_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    correction = await db.get(ClassCorrection, correction_id)
    if not correction:
        raise HTTPException(status_code=404, detail="class correction not found")

    await confirm_unmatched_correction_upload(
        correction,
        db,
        method=request.method,
    )
    await db.refresh(correction)
    return _to_out(correction)


@router.post("/class-corrections/callback", status_code=204)
async def class_correction_callback(data: ClassCorrectionCallbackIn, request: Request):
    token = request.headers.get("x-class-correction-token")
    if token != settings.CLASS_CORRECTION_CALLBACK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid class correction callback token")

    correction_id = uuid.UUID(data.correction_id)
    now = datetime.now(timezone.utc)
    try:
        async with SessionLocal() as db:
            async with db.begin():
                correction = await db.get(ClassCorrection, correction_id, with_for_update=True)
                if not correction:
                    raise HTTPException(status_code=404, detail="class correction not found")

                correction.status = data.status
                correction.upload_bucket = data.upload_bucket or correction.upload_bucket
                correction.upload_image_object_key = data.upload_image_object_key or correction.upload_image_object_key
                correction.upload_label_object_key = data.upload_label_object_key or correction.upload_label_object_key
                correction.result_json = data.result_json
                correction.error_message = data.error_message
                correction.updated_at = now
                if data.status == "processing" and not correction.started_at:
                    correction.started_at = now
                if data.status in ("done", "failed"):
                    correction.finished_at = now
                if data.status == "failed" and data.error_message:
                    await log_error(
                        path="class_corrections:worker",
                        method=request.method,
                        status_code=500,
                        message=data.error_message,
                        stacktrace=None,
                    )
    except HTTPException:
        raise
    except Exception as e:
        await log_error(
            path="class_corrections:callback",
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="class correction callback handling failed") from e
    return
