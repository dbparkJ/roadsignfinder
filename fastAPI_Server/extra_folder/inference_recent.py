"""
최근 추론 결과를 조회해 bbox, class, conf, 원본 파일명, 사용자명을 출력하는 스크립트.

환경변수:
  - DATABASE_URL (default: postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/roadsignfinder)
  - LIMIT (default: 5)

실행:
  python fastAPI_Server/extra_folder/inference_recent.py
"""

import asyncio
import os
import json
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/roadsignfinder",
)
LIMIT = int(os.getenv("LIMIT", "5"))


async def fetch_recent():
    engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                f"""
                select
                  ir.id as job_id,
                  ir.result_json,
                  ir.status,
                  p.original_filename,
                  m.display_name,
                  m.email
                from inference_results ir
                join photos p on ir.photo_id = p.id
                join members m on p.member_id = m.id
                order by ir.created_at desc
                limit {LIMIT}
                """
            )
        )
        rows = result.fetchall()
    await engine.dispose()
    return rows


def parse_selected(result_json):
    if not result_json:
        return None
    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except Exception:
            return None
    selected = result_json.get("selected")
    if not selected or selected == "none":
        return None
    return {
        "bbox": selected.get("xyxy"),
        "class_id": selected.get("class_id"),
        "class_name": selected.get("class_name"),
        "confidence": selected.get("confidence"),
    }


async def main():
    rows = await fetch_recent()
    if not rows:
        print("조회할 결과가 없습니다.")
        return

    for r in rows:
        print(f"raw result_json={r.result_json}")
        selected = parse_selected(r.result_json)
        print(f"job_id={r.job_id}")
        print(f"status={r.status}")
        print(f"user={r.display_name or r.email}")
        print(f"original_filename={r.original_filename}")
        if selected:
            print(f"bbox={selected['bbox']}")
            print(f"class_id={selected['class_id']}, class_name={selected['class_name']}, conf={selected['confidence']}")
        else:
            print("selected: 없음 (no detections)")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
