# app/routes_photos.py
import uuid
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, text

from .deps import get_current_member
from .models import Member, Photo, UploadSession, InferenceJob
from .db import get_db
from .storage import minio_client, MINIO_BUCKET
from .queue_client import celery_client

router = APIRouter()


class PhotoCompleteIn(BaseModel):
    object_key: str


class PhotoCompleteOut(BaseModel):
    ok: bool
    photo_id: str
    job_id: str


@router.post("/photos/complete", response_model=PhotoCompleteOut)
async def complete_upload(
    data: PhotoCompleteIn,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    # 0) 세션이 있어야만 처리(= presign으로 발급된 object_key만 허용)
    r = await db.execute(select(UploadSession).where(UploadSession.object_key == data.object_key))
    sess = r.scalar_one_or_none()
    if not sess:
        raise HTTPException(status_code=404, detail="upload session not found (use /storage/presign first)")

    # 0-1) 내 세션인지 확인
    if sess.member_id != current.id:
        raise HTTPException(status_code=403, detail="not your upload session")

    # 1) 이미 처리된 세션이면 그대로 반환(idempotent)
    if sess.photo_id and sess.job_id:
        return PhotoCompleteOut(ok=True, photo_id=str(sess.photo_id), job_id=str(sess.job_id))

    # 2) MinIO에서 객체 존재/메타 확인
    try:
        st = minio_client.stat_object(MINIO_BUCKET, data.object_key)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"object not found in storage: {e}")

    try:
        # 3) photos upsert 비슷하게: object_key로 이미 있나 확인
        r_photo = await db.execute(select(Photo).where(Photo.object_key == data.object_key))
        photo = r_photo.scalar_one_or_none()

        if photo is None:
            r1 = await db.execute(
                insert(Photo).values(
                    member_id=current.id,
                    object_key=data.object_key,
                    original_filename=sess.original_filename,
                    content_type=sess.content_type or st.content_type,
                    size_bytes=st.size,
                ).returning(Photo.id)
            )
            photo_id = r1.scalar_one()
        else:
            photo_id = photo.id

        # 4) job도 photo_id 기준으로 중복 생성 방지
        r_job = await db.execute(select(InferenceJob).where(InferenceJob.photo_id == photo_id))
        job = r_job.scalar_one_or_none()

        if job is None:
            r2 = await db.execute(
                insert(InferenceJob).values(
                    photo_id=photo_id,
                    member_id=current.id,
                    status="queued",
                    progress=0,
                ).returning(InferenceJob.id)
            )
            job_id = r2.scalar_one()
        else:
            job_id = job.id

        # 5) 세션 갱신
        sess.status = "queued"
        sess.photo_id = photo_id
        sess.job_id = job_id

        await db.commit()

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"DB finalize failed: {e}")

    # 6) 큐 enqueue
    celery_client.send_task("run_inference", args=[str(job_id)], queue="yolo")

    return PhotoCompleteOut(ok=True, photo_id=str(photo_id), job_id=str(job_id))


# ✅ 여기부터 추가: 결과(세그멘테이션) 조회 엔드포인트
@router.get("/photos/{photo_id}/segmentations")
async def get_segmentations(
    photo_id: str,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    # 1) photo_id 유효성 + 내 사진인지 확인
    try:
        pid = uuid.UUID(photo_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid photo_id")

    rp = await db.execute(
        select(Photo).where(Photo.id == pid, Photo.member_id == current.id)
    )
    photo = rp.scalar_one_or_none()
    if not photo:
        raise HTTPException(status_code=404, detail="photo not found")

    # 2) 가장 최근 succeeded job 찾기
    rj = await db.execute(text("""
        SELECT id::text AS job_id
        FROM inference_jobs
        WHERE photo_id = :photo_id
          AND member_id = :member_id
          AND status = 'succeeded'
        ORDER BY finished_at DESC NULLS LAST, created_at DESC
        LIMIT 1
    """), {"photo_id": photo_id, "member_id": str(current.id)})

    rowj = rj.mappings().first()
    if not rowj:
        # 아직 결과 없음(추론중/실패/대기)
        return {"type": "FeatureCollection", "photo_id": photo_id, "job_id": None, "features": []}

    job_id = rowj["job_id"]

    # 3) segmentations를 GeoJSON FeatureCollection으로 반환
    rs = await db.execute(text("""
        SELECT
          id::text AS id,
          class_id,
          class_name,
          score,
          ST_AsGeoJSON(geom_px)::text AS geom_json
        FROM segmentations
        WHERE photo_id = :photo_id
          AND job_id = :job_id
        ORDER BY score DESC
    """), {"photo_id": photo_id, "job_id": job_id})

    features = []
    for m in rs.mappings().all():
        geom = json.loads(m["geom_json"]) if m["geom_json"] else None
        features.append({
            "type": "Feature",
            "id": m["id"],
            "properties": {
                "class_id": m["class_id"],
                "class_name": m["class_name"],
                "score": float(m["score"]) if m["score"] is not None else None,
                "photo_id": photo_id,
                "job_id": job_id,
            },
            "geometry": geom,
        })

    return {
        "type": "FeatureCollection",
        "photo_id": photo_id,
        "job_id": job_id,
        "features": features,
    }
