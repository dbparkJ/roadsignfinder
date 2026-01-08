"""
presign 발급 후 로컬 파일을 업로드하는 간단한 테스트 스크립트.

파일 상단의 변수로 파일 경로와 좌표(img_x, img_y, geo_x, geo_y(UTM 52N))를 지정하거나, 환경변수(FILE_PATH, PRESIGN_IMG_X, PRESIGN_IMG_Y, PRESIGN_UTM_E, PRESIGN_UTM_N)를 사용할 수 있습니다.
예:
  python fastAPI_Server/extra_folder/presign_upload_simple.py
"""

import os
import mimetypes
import uuid

import requests
import time


API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
EMAIL = os.getenv("TEST_EMAIL", "pjmsm0319@naver.com")
PASSWORD = os.getenv("TEST_PASSWORD", "1234")
# 필요 시 여기 값을 직접 수정하세요.
FILE_PATH = os.getenv("FILE_PATH", "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK01/Camera01/Job_20250731_1053_Track12-2-2-2_Sphere_00109.jpg")
IMG_X = float(os.getenv("PRESIGN_IMG_X", 0.0))           # 이미지 내 좌표 x
IMG_Y = float(os.getenv("PRESIGN_IMG_Y", 0.0))           # 이미지 내 좌표 y
# UTM 52N 좌표 (geo_y=Northing, geo_x=Easting)
GEO_X = float(os.getenv("PRESIGN_UTM_E", 322000))
GEO_Y = float(os.getenv("PRESIGN_UTM_N", 4155000))
RDID = os.getenv("PRESIGN_RDID", "TEST-RDID-SIMPLE")

def ensure_file(path: str):
    if not path:
        raise SystemExit("파일 경로를 지정하세요 (FILE_PATH 또는 코드 상단의 FILE_PATH)")
    if not os.path.isfile(path):
        raise SystemExit(f"FILE_PATH가 유효하지 않습니다: {path}")


def ensure_user():
    payload = {"email": EMAIL, "password": PASSWORD, "display_name": "Presign Simple"}
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


def presign(token, filename, content_type, x, y):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(
        f"{API_BASE}/photos/presign",
        headers=headers,
        json={
            "filename": filename,
            "content_type": content_type,
            "img_x": x,
            "img_y": y,
            "geo_x": GEO_X,
            "geo_y": GEO_Y,
            "rdid": RDID,
        },
        timeout=10,
    )
    if r.status_code != 200:
        raise SystemExit(f"[FAIL] presign 실패: {r.status_code} {r.text}")
    data = r.json()
    print(f"[OK] presign 발급: object_key={data['object_key']}")
    return data


def upload(upload_url, content_type, file_path):
    with open(file_path, "rb") as f:
        r = requests.put(upload_url, data=f, headers={"Content-Type": content_type}, timeout=30)
    if r.status_code not in (200, 204):
        raise SystemExit(f"[FAIL] 업로드 실패: {r.status_code} {r.text}")
    print("[OK] 업로드 완료")


def main():
    ensure_file(FILE_PATH)
    #ensure_user()
    token = login()

    filename = os.path.basename(FILE_PATH)
    guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    presign_data = presign(token, filename or f"upload_{uuid.uuid4().hex}", guessed, IMG_X, IMG_Y)
    upload(presign_data["upload_url"], guessed, FILE_PATH)

    session_id = presign_data.get("session_id")
    print(f"[INFO] session_id={session_id} object_key={presign_data['object_key']}")

    # 추론 결과 조회 (간단 폴링)
    if session_id:
        deadline = time.time() + 30
        while time.time() < deadline:
            r = requests.get(f"{API_BASE}/uploads/{session_id}/inference", timeout=10)
            if r.status_code == 200:
                data = r.json()
                print(f"[INFO] 상태: {data['status']}")
                if data["status"] in ("done", "failed"):
                    print("[OK] 최종 결과:", data)
                    break
            else:
                print(f"[WARN] 상태 조회 실패: {r.status_code} {r.text}")
            time.sleep(1)
        else:
            print("[WARN] 제한 시간 내 결과를 받지 못했습니다.")


if __name__ == "__main__":
    main()
