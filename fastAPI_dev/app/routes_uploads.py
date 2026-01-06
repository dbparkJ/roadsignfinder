# app/routes_uploads.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .db import get_db
from .deps import get_current_member
from .models import UploadSession, Member

router = APIRouter(tags=["uploads"])

@router.get("/uploads/{upload_id}")
async def get_upload(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    r = await db.execute(select(UploadSession).where(UploadSession.id == upload_id))
    sess = r.scalar_one_or_none()
    if not sess or sess.member_id != current.id:
        raise HTTPException(status_code=404, detail="not found")

    return {
        "upload_id": str(sess.id),
        "status": sess.status,
        "photo_id": str(sess.photo_id) if sess.photo_id else None,
        "job_id": str(sess.job_id) if sess.job_id else None,
    }
