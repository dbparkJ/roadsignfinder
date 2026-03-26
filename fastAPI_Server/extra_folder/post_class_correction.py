"""
class_corrections POST 테스트 스크립트.

1. RDID로 기존 photo를 찾을 수 있으면:
   JSON 메타데이터만 보내면 worker가 큐에 등록됩니다.

2. RDID로 기존 photo를 찾지 못할 수 있으면:
   서버가 presigned URL을 응답하고, 그 URL로 MinIO에 직접 PUT 하면 됩니다.
   이 스크립트는 FILE_PATH가 있으면 PUT 후 upload-complete까지 호출합니다.

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
    payload = {
        "photo_name": PHOTO_NAME,
        "class_name": CLASS_NAME,
        "rdid": RDID,
        "content_type": mimetypes.guess_type(FILE_PATH)[0] if FILE_PATH else None,
    }

    if FILE_PATH and not os.path.isfile(FILE_PATH):
        raise SystemExit(f"[FAIL] CORRECTION_FILE_PATH가 유효하지 않습니다: {FILE_PATH}")

    response = requests.post(
        f"{API_BASE}/class-corrections",
        json=payload,
        timeout=30,
    )

    print(f"[INFO] status_code={response.status_code}")
    body = None
    try:
        body = response.json()
        print(json.dumps(body, ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)

    if response.status_code != 201:
        raise SystemExit("[FAIL] class correction 등록 실패")

    upload_url = body.get("upload_url") if isinstance(body, dict) else None
    correction_id = body.get("id") if isinstance(body, dict) else None
    if upload_url:
        if not FILE_PATH:
            raise SystemExit("[FAIL] presigned upload_url이 내려왔지만 CORRECTION_FILE_PATH가 없습니다.")
        content_type = mimetypes.guess_type(FILE_PATH)[0] or "application/octet-stream"
        with open(FILE_PATH, "rb") as file_obj:
            upload_resp = requests.put(
                upload_url,
                data=file_obj,
                headers={"Content-Type": content_type},
                timeout=60,
            )
        print(f"[INFO] upload_status_code={upload_resp.status_code}")
        if upload_resp.status_code not in (200, 204):
            raise SystemExit(f"[FAIL] presigned 업로드 실패: {upload_resp.status_code} {upload_resp.text}")

        if correction_id:
            confirm_resp = requests.post(
                f"{API_BASE}/class-corrections/{correction_id}/upload-complete",
                timeout=30,
            )
            print(f"[INFO] confirm_status_code={confirm_resp.status_code}")
            try:
                print(json.dumps(confirm_resp.json(), ensure_ascii=False, indent=2))
            except Exception:
                print(confirm_resp.text)
            if confirm_resp.status_code != 200:
                raise SystemExit("[FAIL] upload-complete 호출 실패")

    print("[OK] class correction 등록 완료")


if __name__ == "__main__":
    main()
