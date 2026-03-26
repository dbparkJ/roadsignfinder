"""
class_corrections POST 테스트 스크립트.

1. RDID로 기존 photo를 찾을 수 있으면:
   photo_name, class_name, rdid만 보내도 worker가 큐에 등록됩니다.

2. RDID로 기존 photo를 찾지 못할 수 있으면:
   같은 요청에 file도 함께 보내면 새 correction bucket으로 업로드됩니다.

실행 예:
  API_BASE=http://localhost:8000 \
  CORRECTION_PHOTO_NAME=sample.jpg \
  CORRECTION_CLASS_NAME=방향표지 \
  CORRECTION_RDID=RDID-1234 \
  python fastAPI_Server/extra_folder/post_class_correction.py

  API_BASE=http://localhost:8000 \
  CORRECTION_PHOTO_NAME=sample.jpg \
  CORRECTION_CLASS_NAME=방향표지 \
  CORRECTION_RDID=UNKNOWN-RDID \
  CORRECTION_FILE_PATH=/path/to/sample.jpg \
  python fastAPI_Server/extra_folder/post_class_correction.py
"""

import json
import mimetypes
import os

import requests


API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
PHOTO_NAME = os.getenv("CORRECTION_PHOTO_NAME", "sample.jpg")
CLASS_NAME = os.getenv("CORRECTION_CLASS_NAME", "방향표지")
RDID = os.getenv("CORRECTION_RDID", "RDID-1234")
FILE_PATH = os.getenv("CORRECTION_FILE_PATH", "").strip()


def main():
    data = {
        "photo_name": PHOTO_NAME,
        "class_name": CLASS_NAME,
        "rdid": RDID,
    }

    files = None
    file_handle = None
    if FILE_PATH:
        if not os.path.isfile(FILE_PATH):
            raise SystemExit(f"[FAIL] CORRECTION_FILE_PATH가 유효하지 않습니다: {FILE_PATH}")
        content_type = mimetypes.guess_type(FILE_PATH)[0] or "application/octet-stream"
        file_handle = open(FILE_PATH, "rb")
        files = {
            "file": (os.path.basename(FILE_PATH), file_handle, content_type),
        }

    try:
        response = requests.post(
            f"{API_BASE}/class-corrections",
            data=data,
            files=files,
            timeout=30,
        )
    finally:
        if file_handle is not None:
            file_handle.close()

    print(f"[INFO] status_code={response.status_code}")
    try:
        body = response.json()
        print(json.dumps(body, ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)

    if response.status_code != 201:
        raise SystemExit("[FAIL] class correction 등록 실패")

    print("[OK] class correction 등록 완료")


if __name__ == "__main__":
    main()
