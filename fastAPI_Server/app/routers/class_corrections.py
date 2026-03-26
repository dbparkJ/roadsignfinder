from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import get_db
from ..models import ClassCorrection
from ..schemas import ClassCorrectionCreateIn, ClassCorrectionOut

router = APIRouter(tags=["class_corrections"])


@router.post("/class-corrections", response_model=ClassCorrectionOut, status_code=201)
async def create_class_correction(
    data: ClassCorrectionCreateIn,
    db: AsyncSession = Depends(get_db),
):
    photo_name = data.photo_name.strip()
    class_name = data.class_name.strip()
    rdid = data.rdid.strip()

    if not photo_name:
        raise HTTPException(status_code=400, detail="photo_name is required")
    if not class_name:
        raise HTTPException(status_code=400, detail="class_name is required")
    if not rdid:
        raise HTTPException(status_code=400, detail="rdid is required")

    correction = ClassCorrection(
        photo_name=photo_name,
        class_name=class_name,
        rdid=rdid,
    )
    db.add(correction)
    await db.commit()
    await db.refresh(correction)

    return ClassCorrectionOut(
        id=str(correction.id),
        photo_name=correction.photo_name,
        class_name=correction.class_name,
        rdid=correction.rdid,
        created_at=correction.created_at,
    )
