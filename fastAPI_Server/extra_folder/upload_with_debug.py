"""
파일 업로드 → 상태/추론 결과 조회 → 에러 로그까지 확인하는 디버그 스크립트.

환경변수 (필요시 수정):
  API_BASE (default: http://localhost:8000)
  TEST_EMAIL (default: debug@example.com)
  TEST_PASSWORD (default: test1234!)
  FILE_PATH (업로드할 파일 경로, 필수)
  IMG_X, IMG_Y, UTM_E, UTM_N, RDID

실행 예:
  FILE_PATH=/path/to/img.jpg IMG_X=0 IMG_Y=0 UTM_E=322000 UTM_N=4155000 RDID=DEBUG-1 \
  python fastAPI_Server/extra_folder/upload_with_debug.py
"""

import os
import mimetypes
import time
import uuid
import requests

API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
EMAIL = os.getenv("TEST_EMAIL", "pjmsm0319@naver.com")
PASSWORD = os.getenv("TEST_PASSWORD", "1234")
FILE_PATH = os.getenv("FILE_PATH", "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK01/Camera01/Job_20250731_1053_Track12-2-2-2_Sphere_00109.jpg")

IMG_X = float(os.getenv("IMG_X", 0.0))
IMG_Y = float(os.getenv("IMG_Y", 0.0))
UTM_E = float(os.getenv("UTM_E", 322000))
UTM_N = float(os.getenv("UTM_N", 4155000))
RDID = os.getenv("RDID", f"DEBUG-{uuid.uuid4().hex[:8]}")


def ensure_file():
    if not FILE_PATH or not os.path.isfile(FILE_PATH):
        raise SystemExit("FILE_PATH 환경변수로 업로드할 파일 경로를 지정하세요.")


def ensure_user():
    payload = {"email": EMAIL, "password": PASSWORD, "display_name": "Debug User"}
    r = requests.post(f"{API_BASE}/auth/register", json=payload, timeout=10)
    if r.status_code in (201, 409):
        print("[OK] 사용자 준비됨")
    else:
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
    data = r.json()
    print(f"[OK] presign 발급: object_key={data['object_key']} session_id={data['session_id']}")
    return data


def upload(upload_url, content_type):
    with open(FILE_PATH, "rb") as f:
        r = requests.put(upload_url, data=f, headers={"Content-Type": content_type}, timeout=60)
    if r.status_code not in (200, 204):
        raise SystemExit(f"[FAIL] 업로드 실패: {r.status_code} {r.text}")
    print("[OK] 업로드 완료")


def wait_inference(job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API_BASE}/inference/{job_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"[INFO] 상태: {data['status']}")
            if data["status"] in ("done", "failed"):
                return data
        else:
            print(f"[WARN] 상태 조회 실패: {r.status_code} {r.text}")
        time.sleep(1)
    return None

def wait_inference_by_session(session_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API_BASE}/uploads/{session_id}/inference", timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"[INFO] 상태: {data['status']}")
            if data["status"] in ("done", "failed"):
                return data
        else:
            print(f"[WARN] 세션 상태 조회 실패: {r.status_code} {r.text}")
        time.sleep(1)
    return None


def main():
    ensure_file()
    ensure_user()
    token = login()

    filename = os.path.basename(FILE_PATH)
    guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    presign_data = presign(token, filename or f"upload_{uuid.uuid4().hex}", guessed)
    upload(presign_data["upload_url"], guessed)

    session_id = presign_data.get("session_id")
    if not session_id:
        print("[WARN] session_id를 presign 응답에서 찾을 수 없습니다.")
        return

    print(f"[INFO] inference 기다리는 중... session_id={session_id}")
    result = wait_inference_by_session(session_id)
    if result:
        print("[OK] 최종 결과:", result)
    else:
        print("[WARN] 타임아웃 내 완료되지 않았습니다.")


if __name__ == "__main__":
    main()
