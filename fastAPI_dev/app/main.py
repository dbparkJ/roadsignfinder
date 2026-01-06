# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone

from .db import get_db, engine, Base
from .models import Member
from .schemas import RegisterIn, LoginIn, TokenOut, MemberOut
from .security import hash_password, verify_password, create_access_token
from .deps import get_current_member

# ✅ routers
from .routes_storage import router as storage_router
from .routes_photos import router as photos_router
from .routes_uploads import router as uploads_router
from .routes_minio_events import router as minio_events_router

from .storage import minio_client, MINIO_BUCKET
from .routes_inference import router as inference_router

print("### LOADED app.main ###")

app = FastAPI(title="JWT Auth + MinIO + Inference")

# ✅ include routers
app.include_router(storage_router)
app.include_router(photos_router)
app.include_router(uploads_router)
app.include_router(minio_events_router)
app.include_router(inference_router)

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

@app.post("/auth/register", response_model=MemberOut, status_code=201)
async def register(data: RegisterIn, db: AsyncSession = Depends(get_db)):
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
