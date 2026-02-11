from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from ..core.db import get_db
from ..models import Member
from ..schemas import RegisterIn, LoginIn, TokenOut, MemberOut
from ..core.security import hash_password, verify_password, create_access_token
from ..core.deps import get_current_member

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=MemberOut, status_code=201)
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


@router.post("/login", response_model=TokenOut)
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


@router.get("/me", response_model=MemberOut)
async def me(current: Member = Depends(get_current_member)):
    return MemberOut(
        id=str(current.id),
        email=current.email,
        display_name=current.display_name,
        role=current.role,
        is_active=current.is_active,
    )
