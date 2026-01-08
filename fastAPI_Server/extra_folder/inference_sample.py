"""
간단한 업로드→추론 결과 확인 샘플.

환경변수:
  API_BASE (default: http://localhost:8000)
  TEST_EMAIL (default: sample@example.com)
  TEST_PASSWORD (default: test1234!)
  FILE_PATH (업로드할 이미지 경로, 필수)
  IMG_X, IMG_Y (기본 0), UTM_E, UTM_N (기본 322000 / 4155000), RDID
"""

import os
import time
import uuid
import mimetypes
import requests

API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
EMAIL = os.getenv("TEST_EMAIL", "pjmsm0319@naver.com")
PASSWORD = os.getenv("TEST_PASSWORD", "1234")
FILE_PATH = os.getenv("FILE_PATH", "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK01/Camera01/Job_20250731_1053_Track12-2-2-2_Sphere_00109.jpg")
IMG_X = float(os.getenv("IMG_X", 750.0))
IMG_Y = float(os.getenv("IMG_Y", 800.0))
UTM_E = float(os.getenv("UTM_E", 322000))
UTM_N = float(os.getenv("UTM_N", 4155000))
RDID = os.getenv("RDID", f"SAMPLE-{uuid.uuid4().hex[:8]}")


def ensure_file():
    if not FILE_PATH or not os.path.isfile(FILE_PATH):
        raise SystemExit("FILE_PATH 환경변수로 업로드할 이미지 경로를 지정하세요.")


def ensure_user():
    payload = {"email": EMAIL, "password": PASSWORD, "display_name": "Sample User"}
    r = requests.post(f"{API_BASE}/auth/register", json=payload, timeout=10)
    if r.status_code not in (201, 409):
        raise SystemExit(f"[FAIL] 회원 준비 실패: {r.status_code} {r.text}")


def login():
    r = requests.post(f"{API_BASE}/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=10)
    if r.status_code != 200:
        raise SystemExit(f"[FAIL] 로그인 실패: {r.status_code} {r.text}")
    return r.json()["access_token"]


def presign(token, filename, content_type):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(
        f"{API_BASE}/photos/presign",
        headers=headers,
        json={
            "filename": filename,
            "content_type": content_type,
            "img_x": IMG_X,
            "img_y": IMG_Y,
            "geo_x": UTM_E,
            "geo_y": UTM_N,
            "rdid": RDID,
        },
        timeout=10,
    )
    if r.status_code != 200:
        raise SystemExit(f"[FAIL] presign 실패: {r.status_code} {r.text}")
    return r.json()


def upload(upload_url, content_type):
    with open(FILE_PATH, "rb") as f:
        r = requests.put(upload_url, data=f, headers={"Content-Type": content_type}, timeout=60)
    if r.status_code not in (200, 204):
        raise SystemExit(f"[FAIL] 업로드 실패: {r.status_code} {r.text}")


def wait_result(session_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API_BASE}/uploads/{session_id}/inference", timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"[INFO] 상태: {data['status']}")
            if data["status"] in ("done", "failed"):
                return data
        else:
            print(f"[WARN] 상태 조회 실패: {r.status_code} {r.text}")
        time.sleep(1)
    return None


def main():
    ensure_file()
    ensure_user()
    token = login()

    filename = os.path.basename(FILE_PATH)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    presign_data = presign(token, filename or f"upload_{uuid.uuid4().hex}", content_type)
    upload(presign_data["upload_url"], content_type)

    session_id = presign_data["session_id"]
    print(f"[INFO] 추론 대기 시작 session_id={session_id}")
    result = wait_result(session_id)
    if result:
        print("[OK] 최종 결과:", result)
    else:
        print("[WARN] 제한 시간 내 결과를 받지 못했습니다.")


if __name__ == "__main__":
    main()
