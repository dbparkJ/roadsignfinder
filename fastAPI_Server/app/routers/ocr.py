import uuid
import traceback
import math
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


def _sanitize_json_payload(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_json_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_payload(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_payload(v) for v in value]
    return value


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
                    "result_json": _sanitize_json_payload(data.result_json),
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
        print(f"[ERROR] ocr_callback failed job_id={data.job_id}: {e}")
        print(traceback.format_exc())
        await log_error(
            path="ocr_callback",
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="ocr callback handling failed") from e
    return
