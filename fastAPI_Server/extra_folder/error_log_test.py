"""
presign 업로드를 일부러 하지 않아 백그라운드에서 오류 로그가 남는지 확인하는 스크립트.

실행 예:
    API_BASE=http://localhost:8000 DATABASE_URL=postgresql+asyncpg://... \
    python fastAPI_Server/extra_folder/error_log_test.py
"""

import os
import time
import uuid

import requests
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
EMAIL = os.getenv("TEST_EMAIL", "pjmsm0319@naver.com")
PASSWORD = os.getenv("TEST_PASSWORD", "1234")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/roadsignfinder",
)


def ensure_user():
    payload = {"email": EMAIL, "password": PASSWORD, "display_name": "Error Tester"}
    r = requests.post(f"{API_BASE}/auth/register", json=payload, timeout=10)
    if r.status_code == 201:
        print("[OK] 사용자 생성 완료")
        return
    if r.status_code == 409:
        print("[OK] 사용자 이미 존재 (409)")
        return
    raise SystemExit(f"[FAIL] 회원가입 실패: {r.status_code} {r.text}")


def login():
    r = requests.post(
        f"{API_BASE}/auth/login",
        json={"email": EMAIL, "password": PASSWORD},
        timeout=10,
    )
    if r.status_code != 200:
        raise SystemExit(f"[FAIL] 로그인 실패: {r.status_code} {r.text}")
    token = r.json()["access_token"]
    print("[OK] 로그인 성공")
    return token


def presign_without_upload(token):
    headers = {"Authorization": f"Bearer {token}"}
    filename = f"error_test_{uuid.uuid4().hex}.png"
    easting = float(os.getenv("TEST_UTM_E", "322000"))
    northing = float(os.getenv("TEST_UTM_N", "4155000"))
    r = requests.post(
        f"{API_BASE}/photos/presign",
        headers=headers,
        json={
            "filename": filename,
            "content_type": "image/png",
            "img_x": 0.0,
            "img_y": 0.0,
            "geo_x": easting,
            "geo_y": northing,
            "rdid": "TEST-RDID-ERROR",
        },
        timeout=10,
    )
    if r.status_code != 200:
        raise SystemExit(f"[FAIL] presign 실패: {r.status_code} {r.text}")
    data = r.json()
    print(f"[OK] presign 발급, object_key={data['object_key']}, 업로드는 생략하여 오류를 유도합니다.")
    return data


async def fetch_error_logs(limit: int = 5):
    engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                select id, path, message, created_at
                from error_logs
                order by created_at desc
                limit :lim
                """
            ),
            {"lim": limit},
        )
        rows = result.fetchall()
    await engine.dispose()
    return rows


def main():
    ensure_user()
    token = login()
    presign_without_upload(token)

    wait_secs = 7
    print(f"[INFO] 업로드 생략 → 백그라운드가 실패 판정할 때까지 {wait_secs}초 대기")
    time.sleep(wait_secs)

    rows = asyncio.run(fetch_error_logs())
    if not rows:
        print("[WARN] error_logs 테이블에 기록이 없습니다.")
        return

    print("[INFO] 최근 오류 로그:")
    for r in rows:
        print(f"- {r.created_at} | {r.path} | {r.message}")


if __name__ == "__main__":
    import asyncio

    main()
