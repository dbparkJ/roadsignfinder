import uuid
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .db import get_db
from .models import Member
from .security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=True)

async def get_current_member(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Member:
    token = creds.credentials
    try:
        payload = decode_access_token(token)
        member_id = payload.get("sub")
        if not member_id:
            raise ValueError("Missing sub")
        mid = uuid.UUID(member_id)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    r = await db.execute(select(Member).where(Member.id == mid))
    member = r.scalar_one_or_none()
    if not member or not member.is_active:
        raise HTTPException(status_code=401, detail="Inactive or missing member")

    return member
