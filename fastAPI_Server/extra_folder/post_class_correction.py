"""
class_corrections POST 테스트 스크립트.

실행 예:
  API_BASE=http://localhost:8000 \
  CORRECTION_PHOTO_NAME=sample.jpg \
  CORRECTION_CLASS_NAME=방향표지 \
  CORRECTION_RDID=RDID-1234 \
  python fastAPI_Server/extra_folder/post_class_correction.py
"""

import os
import json

import requests


API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
PHOTO_NAME = os.getenv("CORRECTION_PHOTO_NAME", "sample.jpg")
CLASS_NAME = os.getenv("CORRECTION_CLASS_NAME", "방향표지")
RDID = os.getenv("CORRECTION_RDID", "RDID-1234")


def main():
    payload = {
        "photo_name": PHOTO_NAME,
        "class_name": CLASS_NAME,
        "rdid": RDID,
    }

    response = requests.post(
        f"{API_BASE}/class-corrections",
        json=payload,
        timeout=10,
    )

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
