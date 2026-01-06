# app/routes_inference.py
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .db import get_db
from .deps import get_current_member
from .models import Member, InferenceJob, Segmentation

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/jobs")
async def list_jobs(
    status: str | None = Query(
        None,
        description="comma separated: queued,running,succeeded,failed (e.g. queued,running)"
    ),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    """
    현재 로그인한 사용자의 inference job 목록 조회.
    큐 모니터링 용도면: /inference/jobs?status=queued,running
    """
    q = (
        select(InferenceJob)
        .where(InferenceJob.member_id == current.id)
        .order_by(InferenceJob.created_at.desc())
        .limit(limit)
    )

    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        if statuses:
            q = q.where(InferenceJob.status.in_(statuses))

    r = await db.execute(q)
    jobs = r.scalars().all()

    return [
        {
            "job_id": str(j.id),
            "photo_id": str(j.photo_id),
            "status": j.status,        # queued/running/succeeded/failed
            "progress": j.progress,    # 0~100
            "error": j.error,
            "created_at": j.created_at,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid job_id")

    r = await db.execute(select(InferenceJob).where(InferenceJob.id == job_uuid))
    job = r.scalar_one_or_none()

    if not job or job.member_id != current.id:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": str(job.id),
        "photo_id": str(job.photo_id),
        "status": job.status,          # queued/running/succeeded/failed
        "progress": job.progress,      # 0~100
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


@router.get("/jobs/{job_id}/results")
async def get_results(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid job_id")

    rj = await db.execute(select(InferenceJob).where(InferenceJob.id == job_uuid))
    job = rj.scalar_one_or_none()
    if not job or job.member_id != current.id:
        raise HTTPException(status_code=404, detail="job not found")

    rr = await db.execute(select(Segmentation).where(Segmentation.job_id == job_uuid))
    rows = rr.scalars().all()

    # QGIS에 바로 먹이는 GeoJSON FeatureCollection
    features = [
        {
            "type": "Feature",
            "geometry": s.geometry,
            "properties": {
                "class_id": s.class_id,
                "class_name": s.class_name,
                "score": s.score,
            },
        }
        for s in rows
    ]

    return {
        "job_id": str(job.id),
        "status": job.status,
        "feature_collection": {"type": "FeatureCollection", "features": features},
    }
