# app/routes_minio_events.py
import os
from urllib.parse import unquote_plus

from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, text

from .db import get_db
from .models import UploadSession, Photo
from .storage import minio_client, MINIO_BUCKET
from .queue_client import celery_client

router = APIRouter(prefix="/storage/internal", tags=["storage-internal"])

WEBHOOK_TOKEN = os.getenv("MINIO_WEBHOOK_TOKEN", "")

@router.post("/minio/events")
async def minio_events(request: Request, db: AsyncSession = Depends(get_db)):
    # (선택) auth_token 검증
    if WEBHOOK_TOKEN:
        auth = request.headers.get("authorization") or ""
        token = auth.replace("Bearer", "").strip()
        if token != WEBHOOK_TOKEN:
            raise HTTPException(status_code=401, detail="invalid webhook token")

    payload = await request.json()
    records = payload.get("Records", [])

    for rec in records:
        event_name = rec.get("eventName", "")
        if not event_name.startswith("s3:ObjectCreated"):
            continue

        key = rec.get("s3", {}).get("object", {}).get("key")
        if not key:
            continue

        object_key = unquote_plus(key)

        # 1) upload_session 찾기
        r = await db.execute(select(UploadSession).where(UploadSession.object_key == object_key))
        sess = r.scalar_one_or_none()
        if not sess:
            continue

        # 2) 이미 처리했으면 스킵
        if sess.photo_id and sess.job_id:
            continue

        # 3) MinIO stat으로 실제 업로드 확인
        st = minio_client.stat_object(MINIO_BUCKET, object_key)

        # 4) Photo 생성
        r1 = await db.execute(
            insert(Photo).values(
                member_id=sess.member_id,
                object_key=object_key,
                original_filename=sess.original_filename,
                content_type=sess.content_type or st.content_type,
                size_bytes=st.size,
            ).returning(Photo.id)
        )
        photo_id = r1.scalar_one()

        # 5) Job 생성
        r2 = await db.execute(text("""
            INSERT INTO inference_jobs(photo_id, member_id, status, progress)
            VALUES(:photo_id, :member_id, 'queued', 0)
            RETURNING id
        """), {"photo_id": str(photo_id), "member_id": str(sess.member_id)})
        job_id = r2.scalar_one()

        # 6) 세션 업데이트
        sess.status = "queued"
        sess.photo_id = photo_id
        sess.job_id = job_id
        await db.commit()

        # 7) 큐 enqueue
        celery_client.send_task("run_inference", args=[str(job_id)], queue="yolo")

    return {"ok": True}
