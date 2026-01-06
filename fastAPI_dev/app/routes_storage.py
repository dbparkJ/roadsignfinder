# app/routes_storage.py
import uuid
from datetime import timedelta
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from .deps import get_current_member
from .db import get_db
from .models import Member, UploadSession
from .storage import minio_client, MINIO_BUCKET

router = APIRouter()

class PresignIn(BaseModel):
    filename: str
    content_type: str

class PresignOut(BaseModel):
    upload_id: str
    object_key: str
    upload_url: str

@router.post("/storage/presign", response_model=PresignOut)
async def presign_upload(
    data: PresignIn,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    ext = ""
    if "." in data.filename:
        ext = "." + data.filename.rsplit(".", 1)[1].lower()

    upload_id = str(uuid.uuid4())
    object_key = f"members/{current.id}/uploads/{uuid.uuid4()}{ext}"

    # ✅ 업로드 세션 미리 기록
    await db.execute(
        insert(UploadSession).values(
            id=upload_id,
            member_id=current.id,
            object_key=object_key,
            original_filename=data.filename,
            content_type=data.content_type,
            status="issued",
        )
    )
    await db.commit()

    upload_url = minio_client.presigned_put_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_key,
        expires=timedelta(minutes=15),
    )
    return PresignOut(upload_id=upload_id, object_key=object_key, upload_url=upload_url)
