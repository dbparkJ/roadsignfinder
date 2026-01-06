"""
presign 업로드 → MinIO PUT → 백그라운드 DB 기록 여부를 점검하는 단일 스크립트.

실행 예:
    API_BASE=http://localhost:8000 DATABASE_URL=postgresql+asyncpg://... \
    python fastAPI_Server/extra_folder/presign_flow_test.py
"""

import asyncio
import base64
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

# 1x1 PNG (투명) 샘플
PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/xcAAn8B9pU56vQAAAAASUVORK5CYII="
)


def ensure_user():
    payload = {"email": EMAIL, "password": PASSWORD, "display_name": "Presign Tester"}
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


def presign(token):
    headers = {"Authorization": f"Bearer {token}"}
    filename = f"test_{uuid.uuid4().hex}.png"
    # UTM 52N 예시 좌표 (geo_lat=Northing, geo_lng=Easting)
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
            "rdid": "TEST-RDID-1234",
        },
        timeout=10,
    )
    if r.status_code != 200:
        raise SystemExit(f"[FAIL] presign 실패: {r.status_code} {r.text}")
    data = r.json()
    print(f"[OK] presign 발급: object_key={data['object_key']}")
    return data


def put_presigned(upload_url):
    r = requests.put(
        upload_url,
        data=PNG_BYTES,
        headers={"Content-Type": "image/png"},
        timeout=10,
    )
    if r.status_code not in (200, 204):
        raise SystemExit(f"[FAIL] presigned 업로드 실패: {r.status_code} {r.text}")
    print("[OK] presigned URL로 업로드 완료")


async def fetch_photo(member_id: str, object_key: str):
    engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                select id, size_bytes, created_at
                from photos
                where member_id = :mid and object_key = :obj
                order by created_at desc
                limit 1
                """
            ),
            {"mid": member_id, "obj": object_key},
        )
        row = result.fetchone()
    await engine.dispose()
    return row

async def fetch_upload_session(session_id: str):
    engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                select status, photo_id, uploaded_at
                from upload_sessions
                where id = :sid
                """
            ),
            {"sid": session_id},
        )
        row = result.fetchone()
    await engine.dispose()
    return row


def main():
    #ensure_user()
    token = login()

    presign_data = presign(token)
    upload_url = presign_data["upload_url"]
    object_key = presign_data["object_key"]
    member_id = object_key.split("/")[0]
    session_id = presign_data.get("session_id")

    put_presigned(upload_url)

    print("[INFO] 백그라운드 기록을 기다리는 중...")
    for i in range(10):
        photo_row = asyncio.run(fetch_photo(member_id, object_key))
        session_row = asyncio.run(fetch_upload_session(session_id)) if session_id else None

        if session_row:
            print(f"[INFO] 세션 상태: status={session_row.status}, photo_id={session_row.photo_id}, uploaded_at={session_row.uploaded_at}")

        if photo_row:
            print(f"[OK] DB 기록 확인: id={photo_row.id}, size={photo_row.size_bytes}, created_at={photo_row.created_at}")
            break
        time.sleep(1)
    else:
        raise SystemExit("[FAIL] DB에 사진 메타데이터가 기록되지 않았습니다.")


if __name__ == "__main__":
    main()
