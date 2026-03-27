import uuid
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.db import get_db, SessionLocal
from ..models import InferenceResult, PoleTypeDebugLog, PoleTypeResultCache, UploadSession
from ..schemas import PoleTypeResultOut, PoleTypeCallbackIn
from ..services.upload import (
    log_error,
    resolve_inference_status,
    update_upload_sessions_status_by_job,
)

router = APIRouter(tags=["pole_type"])


@router.post("/sam3/callback", status_code=204, include_in_schema=False)
@router.post("/pole_type/callback", status_code=204)
async def pole_type_callback(data: PoleTypeCallbackIn, request: Request):
    token = request.headers.get("x-pole-type-token")
    if token != settings.POLE_TYPE_CALLBACK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid pole_type callback token")

    job_id = uuid.UUID(data.job_id)
    now = datetime.now(timezone.utc)
    try:
        async with SessionLocal() as db:
            async with db.begin():
                job = await db.get(InferenceResult, job_id, with_for_update=True)
                if not job:
                    raise HTTPException(status_code=404, detail="inference job not found for pole_type callback")

                merged = dict(job.result_json or {})
                pole_type_payload = {
                    "status": data.status,
                    "result_object_key": data.result_object_key,
                    "result_json": data.result_json,
                    "error_message": data.error_message,
                    "size_bytes": data.size_bytes,
                    "updated_at": now.isoformat(),
                }
                if data.status == "processing" and not (merged.get("pole_type") or {}).get("started_at"):
                    pole_type_payload["started_at"] = now.isoformat()
                if data.status in ("done", "failed"):
                    pole_type_payload["finished_at"] = now.isoformat()
                merged["pole_type"] = pole_type_payload
                job.result_json = merged
                job.status = resolve_inference_status(job.result_json, fallback=job.status) or job.status
                if job.status in ("done", "failed"):
                    job.finished_at = now
                job.updated_at = now

                if data.status in ("done", "failed"):
                    db.add(
                        PoleTypeResultCache(
                            photo_id=job.photo_id,
                            status=data.status,
                            result_object_key=data.result_object_key,
                            result_json=data.result_json,
                            error_message=data.error_message,
                        )
                    )

                if settings.POLE_TYPE_DEBUG_LOG:
                    db.add(
                        PoleTypeDebugLog(
                            inference_job_id=job.id,
                            photo_id=job.photo_id,
                            status=pole_type_payload.get("status"),
                            result_object_key=pole_type_payload.get("result_object_key"),
                            result_json=pole_type_payload.get("result_json"),
                            error_message=pole_type_payload.get("error_message"),
                        )
                    )

                await update_upload_sessions_status_by_job(job.id, db)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] pole_type_callback failed job_id={data.job_id}: {e}")
        print(traceback.format_exc())
        await log_error(
            path="pole_type_callback",
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="pole_type callback handling failed") from e
    return


@router.get("/pole_type/{job_id}", response_model=PoleTypeResultOut)
async def get_pole_type_result(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(InferenceResult).where(InferenceResult.id == job_id))
    job = r.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="inference job not found")
    pole_type = (job.result_json or {}).get("pole_type") if isinstance(job.result_json, dict) else None
    if not pole_type:
        raise HTTPException(status_code=404, detail="pole_type result not found")
    return PoleTypeResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=pole_type.get("status") or "pending",
        result_object_key=pole_type.get("result_object_key"),
        result_json=pole_type.get("result_json"),
        error_message=pole_type.get("error_message"),
        rdid=job.rdid,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=pole_type.get("started_at"),
        finished_at=pole_type.get("finished_at"),
        size_bytes=pole_type.get("size_bytes"),
    )


@router.get("/pole_type/result", response_model=PoleTypeResultOut)
async def get_pole_type_result_generic(
    job_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    photo_id: uuid.UUID | None = None,
    rdid: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    if not any([job_id, session_id, photo_id, rdid]):
        raise HTTPException(status_code=400, detail="job_id, session_id, photo_id, rdid 중 하나는 필요합니다.")

    job: InferenceResult | None = None

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
        raise HTTPException(status_code=404, detail="inference job not found")

    pole_type = (job.result_json or {}).get("pole_type") if isinstance(job.result_json, dict) else None
    if not pole_type:
        raise HTTPException(status_code=404, detail="pole_type result not found")

    return PoleTypeResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=pole_type.get("status") or "pending",
        result_object_key=pole_type.get("result_object_key"),
        result_json=pole_type.get("result_json"),
        error_message=pole_type.get("error_message"),
        rdid=job.rdid,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=pole_type.get("started_at"),
        finished_at=pole_type.get("finished_at"),
        size_bytes=pole_type.get("size_bytes"),
    )


@router.get("/uploads/{session_id}/pole_type", response_model=PoleTypeResultOut)
async def get_pole_type_by_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    us = await db.execute(select(UploadSession).where(UploadSession.id == session_id))
    upload_session = us.scalar_one_or_none()
    if not upload_session:
        raise HTTPException(status_code=404, detail="upload session not found")
    if not upload_session.job_id:
        return PoleTypeResultOut(
            id=str(upload_session.id),
            photo_id=str(upload_session.photo_id) if upload_session.photo_id else "",
            status=upload_session.status,
            result_object_key=None,
            result_json=None,
            error_message="inference job not created yet",
            created_at=upload_session.created_at,
            updated_at=upload_session.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    job = await db.get(InferenceResult, upload_session.job_id)
    if not job:
        return PoleTypeResultOut(
            id=str(upload_session.job_id),
            photo_id=str(upload_session.photo_id) if upload_session.photo_id else "",
            status="pending",
            result_object_key=None,
            result_json=None,
            error_message="inference job not found yet",
            created_at=upload_session.created_at,
            updated_at=upload_session.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    pole_type = (job.result_json or {}).get("pole_type") if isinstance(job.result_json, dict) else None
    if not pole_type:
        return PoleTypeResultOut(
            id=str(job.id),
            photo_id=str(job.photo_id),
            status="pending",
            result_object_key=None,
            result_json=None,
            error_message="pole_type result not found yet",
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    return PoleTypeResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=pole_type.get("status") or "pending",
        result_object_key=pole_type.get("result_object_key"),
        result_json=pole_type.get("result_json"),
        error_message=pole_type.get("error_message"),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=pole_type.get("started_at"),
        finished_at=pole_type.get("finished_at"),
        size_bytes=pole_type.get("size_bytes"),
    )
