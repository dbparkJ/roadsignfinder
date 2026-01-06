"""
최근 inference_callback 에러 로그를 조회하는 스크립트.

환경변수:
  - DATABASE_URL (기본: postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/roadsignfinder)

실행:
  python fastAPI_Server/extra_folder/error_log_fetch.py
"""

import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/roadsignfinder")


async def main():
    engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                select created_at, message, stacktrace
                from error_logs
                where path = 'inference_callback'
                order by created_at desc
                limit 5
                """
            )
        )
        rows = result.fetchall()
        if not rows:
            print("로그가 없습니다.")
            return
        for r in rows:
            print(f"[{r.created_at}] {r.message}")
            if r.stacktrace:
                print(r.stacktrace)
                print("-" * 40)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
