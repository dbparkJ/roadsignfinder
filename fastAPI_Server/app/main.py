# app/main.py
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.concurrency import run_in_threadpool
from minio.error import S3Error
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone, timedelta
import traceback
from celery import Celery

from .db import get_db, engine, Base, SessionLocal
from .models import Member, Photo, UploadSession, ErrorLog, InferenceResult
from .schemas import (
    RegisterIn,
    LoginIn,
    TokenOut,
    MemberOut,
    PhotoOut,
    PhotoPresignIn,
    PhotoPresignOut,
    InferenceResultOut,
    InferenceCallbackIn,
)
from .security import hash_password, verify_password, create_access_token
from .deps import get_current_member
from .storage import minio_client, MINIO_BUCKET
from .config import settings

print("### LOADED app.main ###")

app = FastAPI(title="JWT Auth + MinIO + Inference")
celery_app = Celery(
    "inference_worker",
    broker=settings.CELERY_BROKER_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="inference",
)

def _normalize_dt(dt):
    if not dt:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

async def log_error(path: str | None, method: str | None, status_code: int | None, message: str | None, stacktrace: str | None = None):
    async with SessionLocal() as session:
        try:
            session.add(
                ErrorLog(
                    path=path,
                    method=method,
                    status_code=status_code,
                    message=message,
                    stacktrace=stacktrace,
                )
            )
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"[WARN] error log 저장 실패: {e}")

async def schedule_inference(photo: Photo, db: AsyncSession) -> InferenceResult:
    job = InferenceResult(photo_id=photo.id, status="queued", rdid=photo.rdid)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    payload = {
        "job_id": str(job.id),
        "photo_id": str(photo.id),
        "bucket": MINIO_BUCKET,
        "object_key": photo.object_key,
        "rdid": photo.rdid,
        "img_x": photo.img_x,
        "img_y": photo.img_y,
    }
    try:
        await run_in_threadpool(
            celery_app.send_task,
            "inference_worker.tasks.run_inference",
            args=[],
            kwargs=payload,
            queue="inference",
        )
    except Exception as e:
        job.status = "failed"
        job.error_message = f"enqueue failed: {e}"
        await db.commit()
        await log_error(
            path="enqueue_inference",
            method=None,
            status_code=None,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail="추론 작업 생성에 실패했습니다.") from e

    return job

async def _register_uploaded_photo(session_id, member_id, object_key, original_filename, content_type):
    """
    presigned 업로드 완료 후 MinIO에 객체가 생기면 DB에 기록한다.
    presign 발급 시 백그라운드 태스크로 호출된다.
    """
    async with SessionLocal() as session:
        async def _update_status(status: str):
            try:
                us = await session.execute(select(UploadSession).where(UploadSession.id == session_id))
                upload_session = us.scalar_one_or_none()
                if upload_session:
                    upload_session.status = status
                    await session.commit()
            except Exception as e:
                await session.rollback()
                print(f"[WARN] upload_session 상태 업데이트 실패(session_id={session_id}, status={status}): {e}")

        for attempt in range(5):
            try:
                stat = await run_in_threadpool(minio_client.stat_object, MINIO_BUCKET, object_key)
                photo = None
                existing = await session.execute(select(Photo).where(Photo.object_key == object_key))
                photo = existing.scalar_one_or_none()
                if not photo:
                    us_for_xy = await session.execute(select(UploadSession).where(UploadSession.id == session_id))
                    xy = us_for_xy.scalar_one_or_none()
                    photo = Photo(
                        member_id=member_id,
                        object_key=object_key,
                        original_filename=original_filename,
                        content_type=content_type,
                        img_x=xy.img_x if xy else 0.0,
                        img_y=xy.img_y if xy else 0.0,
                        geo_x=xy.geo_x if xy else 0.0,
                        geo_y=xy.geo_y if xy else 0.0,
                        geo_point=f"SRID=32652;POINT({xy.geo_x if xy else 0.0} {xy.geo_y if xy else 0.0})",
                        rdid=xy.rdid if xy else None,
                        size_bytes=stat.size,
                        created_at=_normalize_dt(stat.last_modified),
                    )
                    session.add(photo)
                    await session.flush()

                us = await session.execute(select(UploadSession).where(UploadSession.id == session_id))
                upload_session = us.scalar_one_or_none()
                if upload_session:
                    upload_session.status = "processing"
                    upload_session.photo_id = photo.id
                    upload_session.uploaded_at = _normalize_dt(stat.last_modified)

                await session.commit()
                try:
                    job = await schedule_inference(photo, session)
                    if upload_session:
                        upload_session.job_id = job.id
                        upload_session.status = "queued"
                        await session.commit()
                except Exception:
                    await session.rollback()
                    await log_error(
                        path="enqueue_inference",
                        method=None,
                        status_code=None,
                        message="failed to enqueue inference",
                        stacktrace=traceback.format_exc(),
                    )
                return
            except S3Error as e:
                if e.code == "NoSuchKey":
                    await asyncio.sleep(1)
                    continue
                print(f"[WARN] MinIO stat_object 실패(object_key={object_key}): {e}")
                await _update_status("failed")
                await log_error(
                    path="background:_register_uploaded_photo",
                    method=None,
                    status_code=None,
                    message=f"S3Error: {e}",
                    stacktrace=None,
                )
                return
            except Exception as e:
                await session.rollback()
                print(f"[WARN] presign 업로드 DB 기록 실패(object_key={object_key}): {e}")
                await _update_status("failed")
                await log_error(
                    path="background:_register_uploaded_photo",
                    method=None,
                    status_code=None,
                    message=str(e),
                    stacktrace=traceback.format_exc(),
                )
                return
        print(f"[WARN] presign 업로드 확인 실패(object_key={object_key})")
        await _update_status("missing")
        await log_error(
            path="background:_register_uploaded_photo",
            method=None,
            status_code=None,
            message="presigned upload not found in MinIO",
            stacktrace=None,
        )

@app.on_event("startup")
async def startup():
    # (운영은 Alembic 권장)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ✅ MinIO 버킷 자동 생성 (없으면 생성) - 연결 문제로 앱이 죽지 않게 방어
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
            print(f"[OK] Created bucket: {MINIO_BUCKET}")
    except Exception as e:
        print(f"[WARN] MinIO bucket check/create failed: {e}")

@app.get("/health")
def health():
    return {"ok": True}

@app.middleware("http")
async def log_unhandled_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        await log_error(
            path=str(request.url),
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise

@app.post("/auth/register", response_model=MemberOut, status_code=201)
async def register(data: RegisterIn, db: AsyncSession = Depends(get_db)):
    """
    회원가입 엔드포인트

    현 버전에서는 관리자만 가입 가능하게 구현하였으며, 관리 코드는 extra_folder에 있음

    1. 비밀번호 해싱
    2. DB에 회원 정보 저장
    3. 중복 이메일 처리
    4. 회원 정보 반환
    """
    m = Member(
        email=str(data.email),
        password_hash=hash_password(data.password),
        display_name=data.display_name,
        role="user",
        is_active=True,
    )
    db.add(m)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email already exists")

    await db.refresh(m)
    return MemberOut(
        id=str(m.id),
        email=m.email,
        display_name=m.display_name,
        role=m.role,
        is_active=m.is_active,
    )

@app.post("/auth/login", response_model=TokenOut)
async def login(data: LoginIn, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(Member).where(Member.email == str(data.email)))
    m = r.scalar_one_or_none()

    if not m or not m.is_active:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(data.password, m.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    m.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token(sub=str(m.id), role=m.role)
    return TokenOut(access_token=token)

@app.get("/auth/me", response_model=MemberOut)
async def me(current: Member = Depends(get_current_member)):
    return MemberOut(
        id=str(current.id),
        email=current.email,
        display_name=current.display_name,
        role=current.role,
        is_active=current.is_active,
    )

@app.post("/photos/presign", response_model=PhotoPresignOut)
async def presign_photo_upload(
    data: PhotoPresignIn,
    db: AsyncSession = Depends(get_db),
    current: Member = Depends(get_current_member),
):
    if not data.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 비어 있습니다.")
    if data.geo_x is None or data.geo_y is None:
        raise HTTPException(status_code=400, detail="geo_x, geo_y 좌표는 필수입니다.")
    if data.img_x is None or data.img_y is None:
        raise HTTPException(status_code=400, detail="img_x, img_y 좌표는 필수입니다.")
    if not data.rdid:
        raise HTTPException(status_code=400, detail="rdid는 필수입니다.")

    ext = Path(data.filename).suffix.lower()
    object_key = f"{current.id}/{uuid.uuid4().hex}{ext}"
    expires_in = 600  # 10분
    content_type = data.content_type or "application/octet-stream"

    upload_session = UploadSession(
        member_id=current.id,
        object_key=object_key,
        original_filename=data.filename,
        content_type=content_type,
        status="issued",
        img_x=data.img_x,
        img_y=data.img_y,
        geo_x=data.geo_x,
        geo_y=data.geo_y,
        geo_point=f"SRID=32652;POINT({data.geo_x} {data.geo_y})",
        rdid=data.rdid,
    )
    db.add(upload_session)
    try:
        await db.commit()
        await db.refresh(upload_session)
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=500, detail="업로드 세션 생성에 실패했습니다.")

    try:
        upload_url = await run_in_threadpool(
            minio_client.presigned_put_object,
            MINIO_BUCKET,
            object_key,
            expires=timedelta(seconds=expires_in),
        )
    except Exception as e:
        try:
            upload_session.status = "failed"
            await db.commit()
        except Exception:
            await db.rollback()
        raise HTTPException(status_code=502, detail=f"presigned URL 생성 실패: {e}") from e

    # 업로드 완료 후 DB 기록/추론 큐잉을 비동기로 처리
    asyncio.create_task(
        _register_uploaded_photo(
            session_id=upload_session.id,
            member_id=current.id,
            object_key=object_key,
            original_filename=data.filename,
            content_type=content_type,
        )
    )

    return PhotoPresignOut(
        session_id=str(upload_session.id),
        upload_url=upload_url,
        object_key=object_key,
        bucket=MINIO_BUCKET,
        expires_in=expires_in,
    )

@app.post("/photos", response_model=PhotoOut, status_code=201)
async def upload_photo(
    file: UploadFile = File(...),
    img_x: float = Form(...),
    img_y: float = Form(...),
    geo_x: float = Form(...),
    geo_y: float = Form(...),
    rdid: str = Form(...),
    current: Member = Depends(get_current_member),
    db: AsyncSession = Depends(get_db),
):
    if not rdid:
        raise HTTPException(status_code=400, detail="rdid는 필수입니다.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 비어 있습니다.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    ext = Path(file.filename).suffix.lower()
    object_key = f"{current.id}/{uuid.uuid4().hex}{ext}"
    content_type = file.content_type or "application/octet-stream"

    try:
        await run_in_threadpool(
            minio_client.put_object,
            MINIO_BUCKET,
            object_key,
            file.file,
            -1,
            10 * 1024 * 1024,
            content_type=content_type,
        )
    except S3Error as e:
        raise HTTPException(status_code=502, detail=f"파일 업로드 실패: {e.code}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail="파일 업로드 중 오류가 발생했습니다.") from e

    photo = Photo(
        member_id=current.id,
        object_key=object_key,
        original_filename=file.filename,
        content_type=content_type,
        size_bytes=None,
        img_x=img_x,
        img_y=img_y,
        geo_x=geo_x,
        geo_y=geo_y,
        geo_point=f"SRID=32652;POINT({geo_x} {geo_y})",
        rdid=rdid,
    )

    db.add(photo)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        try:
            await run_in_threadpool(minio_client.remove_object, MINIO_BUCKET, object_key)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="업로드 메타데이터 저장에 실패했습니다.")

    await db.refresh(photo)

    try:
        stat = await run_in_threadpool(minio_client.stat_object, MINIO_BUCKET, object_key)
        photo.size_bytes = stat.size
        await db.commit()
        await db.refresh(photo)
    except Exception:
        pass

    try:
        await schedule_inference(photo, db)
    except Exception:
        pass

    return PhotoOut(
        id=str(photo.id),
        object_key=photo.object_key,
        bucket=MINIO_BUCKET,
        original_filename=photo.original_filename,
        content_type=photo.content_type,
        size_bytes=photo.size_bytes,
        created_at=photo.created_at,
        img_x=photo.img_x,
        img_y=photo.img_y,
        geo_x=photo.geo_x,
        geo_y=photo.geo_y,
        rdid=photo.rdid,
    )

@app.post("/inference/callback", status_code=204)
async def inference_callback(data: InferenceCallbackIn, request: Request):
    token = request.headers.get("x-inference-token")
    if token != settings.INFERENCE_CALLBACK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid callback token")

    job_id = uuid.UUID(data.job_id)
    now = datetime.now(timezone.utc)
    try:
        async with SessionLocal() as db:
            async with db.begin():
                job = await db.get(InferenceResult, job_id, with_for_update=True)
                if not job:
                    raise HTTPException(status_code=404, detail="job not found")

                job.status = data.status
                job.result_object_key = data.result_object_key
                job.result_json = data.result_json
                job.error_message = data.error_message
                job.size_bytes = data.size_bytes
                if data.status == "processing" and not job.started_at:
                    job.started_at = now
                if data.status in ("done", "failed"):
                    job.finished_at = now
                job.updated_at = now
    except HTTPException:
        raise
    except Exception as e:
        await log_error(
            path="inference_callback",
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="callback handling failed") from e
    return

@app.get("/inference/{job_id}", response_model=InferenceResultOut)
async def get_inference_result(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(InferenceResult).where(InferenceResult.id == job_id))
    job = r.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return InferenceResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=job.status,
        result_object_key=job.result_object_key,
        result_json=job.result_json,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        size_bytes=job.size_bytes,
    )

@app.get("/inference/result", response_model=InferenceResultOut)
async def get_inference_result_generic(
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
        job = q.scalar_one_or_none()
    elif rdid:
        q = await db.execute(
            select(InferenceResult).where(InferenceResult.rdid == rdid).order_by(InferenceResult.created_at.desc())
        )
        job = q.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="inference result not found")

    return InferenceResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=job.status,
        result_object_key=job.result_object_key,
        result_json=job.result_json,
        error_message=job.error_message,
        rdid=job.rdid,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        size_bytes=job.size_bytes,
    )

@app.get("/uploads/{session_id}/inference", response_model=InferenceResultOut)
async def get_inference_by_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    us = await db.execute(select(UploadSession).where(UploadSession.id == session_id))
    upload_session = us.scalar_one_or_none()
    if not upload_session:
        raise HTTPException(status_code=404, detail="upload session not found")
    if not upload_session.job_id:
        return InferenceResultOut(
            id=str(upload_session.id),
            photo_id=str(upload_session.photo_id) if upload_session.photo_id else "",
            status=upload_session.status,
            result_object_key=None,
            result_json=None,
            error_message="job not created yet",
            created_at=upload_session.created_at,
            updated_at=upload_session.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    job = await db.get(InferenceResult, upload_session.job_id)
    if not job:
        return InferenceResultOut(
            id=str(upload_session.job_id),
            photo_id=str(upload_session.photo_id) if upload_session.photo_id else "",
            status="pending",
            result_object_key=None,
            result_json=None,
            error_message="job not found yet",
            created_at=upload_session.created_at,
            updated_at=upload_session.updated_at,
            started_at=None,
            finished_at=None,
            size_bytes=None,
        )

    return InferenceResultOut(
        id=str(job.id),
        photo_id=str(job.photo_id),
        status=job.status,
        result_object_key=job.result_object_key,
        result_json=job.result_json,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        size_bytes=job.size_bytes,
    )

@app.get("/inference/{job_id}/wait", response_model=InferenceResultOut)
async def wait_inference_result(job_id: uuid.UUID, timeout_seconds: int = 30, db: AsyncSession = Depends(get_db)):
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
    while True:
        r = await db.execute(select(InferenceResult).where(InferenceResult.id == job_id))
        job = r.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status in ("done", "failed"):
            return InferenceResultOut(
                id=str(job.id),
                photo_id=str(job.photo_id),
                status=job.status,
                result_object_key=job.result_object_key,
                result_json=job.result_json,
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        if datetime.now(timezone.utc) >= deadline:
            return InferenceResultOut(
                id=str(job.id),
                photo_id=str(job.photo_id),
                status=job.status,
                result_object_key=job.result_object_key,
                result_json=job.result_json,
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        await asyncio.sleep(1)
