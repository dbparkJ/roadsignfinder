import uuid
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from ..core.config import settings
from ..core.db import SessionLocal
from ..models import InferenceResult, UploadSession
from ..schemas import OcrCallbackIn
from ..services.upload import (
    log_error,
    update_upload_session_status,
    resolve_inference_status,
)

router = APIRouter(tags=["ocr"])


@router.post("/ocr/callback", status_code=204)
async def ocr_callback(data: OcrCallbackIn, request: Request):
    token = request.headers.get("x-ocr-token")
    if token != settings.OCR_CALLBACK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid ocr callback token")

    job_id = uuid.UUID(data.job_id)
    now = datetime.now(timezone.utc)
    try:
        async with SessionLocal() as db:
            async with db.begin():
                job = await db.get(InferenceResult, job_id, with_for_update=True)
                if not job:
                    raise HTTPException(status_code=404, detail="inference job not found for ocr callback")

                merged = dict(job.result_json or {})
                ocr_payload = {
                    "status": data.status,
                    "result_json": data.result_json,
                    "error_message": data.error_message,
                    "size_bytes": data.size_bytes,
                    "updated_at": now.isoformat(),
                }
                if data.status == "processing" and not (merged.get("ocr") or {}).get("started_at"):
                    ocr_payload["started_at"] = now.isoformat()
                if data.status in ("done", "failed"):
                    ocr_payload["finished_at"] = now.isoformat()

                merged["ocr"] = ocr_payload
                job.result_json = merged
                job.status = resolve_inference_status(job.result_json, fallback=job.status) or job.status
                if job.status in ("done", "failed"):
                    job.finished_at = now
                job.updated_at = now

                us = await db.execute(select(UploadSession).where(UploadSession.job_id == job.id))
                upload_session = us.scalar_one_or_none()
                if upload_session:
                    await update_upload_session_status(upload_session, db)
    except HTTPException:
        raise
    except Exception as e:
        await log_error(
            path="ocr_callback",
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="ocr callback handling failed") from e
    return
