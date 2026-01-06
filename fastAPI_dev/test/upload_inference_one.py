import os
import mimetypes
import requests

# =========================
# Config
# =========================
API_BASE = "http://111.111.111.164:8000"
EMAIL = "pjmsm0319@naver.com"
PASSWORD = "1234"

INPUT_DIR = r"D:\1.표지판 인식\3.학습자료\test"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# =========================
# Utilities
# =========================
def iter_images(input_dir: str, allowed_ext: set[str] = ALLOWED_EXT):
    for root, _, files in os.walk(input_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in allowed_ext:
                yield os.path.join(root, fn)


def guess_content_type(path: str) -> str:
    ct, _ = mimetypes.guess_type(path)
    return ct or "application/octet-stream"


# =========================
# API wrappers
# =========================
def api_login(api_base: str, email: str, password: str) -> dict:
    r = requests.post(
        f"{api_base}/auth/login",
        json={"email": email, "password": password},
        timeout=20,
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def api_presign(api_base: str, headers: dict, filename: str, content_type: str) -> tuple[str, str]:
    """
    return: (upload_id, upload_url)
    ※ 개발단계에서는 upload_id도 굳이 안 써도 되지만,
      presign 응답에 들어있어서 그냥 받아둠.
    """
    r = requests.post(
        f"{api_base}/storage/presign",
        json={"filename": filename, "content_type": content_type},
        headers=headers,
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    return j["upload_id"], j["upload_url"]


def api_upload_to_presigned_url(upload_url: str, filepath: str, content_type: str) -> None:
    with open(filepath, "rb") as f:
        r = requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": content_type},
            timeout=300,
        )
    r.raise_for_status()


# =========================
# Minimal pipeline (No tracking)
# =========================
def upload_images_only(
    api_base: str = API_BASE,
    email: str = EMAIL,
    password: str = PASSWORD,
    input_dir: str = INPUT_DIR,
    allowed_ext: set[str] = ALLOWED_EXT,
) -> dict:
    """
    개발단계용: 추적/폴링 없이 업로드만 수행.
    - 서버가 업로드 이후 job을 만들든 말든 클라이언트는 신경 안 씀.
    - 성공/실패는 '업로드(PUT) 성공/실패'만 기준으로 판단.

    return:
    {
      "total_files": int,
      "uploaded": int,
      "failed": int,
      "failed_files": [{"filepath":..., "error":...}, ...]
    }
    """
    headers = api_login(api_base, email, password)

    paths = list(iter_images(input_dir, allowed_ext))
    result = {
        "total_files": len(paths),
        "uploaded": 0,
        "failed": 0,
        "failed_files": [],
    }

    for p in paths:
        filename = os.path.basename(p)
        content_type = guess_content_type(p)

        try:
            _, upload_url = api_presign(api_base, headers, filename, content_type)
            api_upload_to_presigned_url(upload_url, p, content_type)
            result["uploaded"] += 1
        except Exception as e:
            result["failed"] += 1
            result["failed_files"].append({"filepath": p, "error": str(e)})

    return result


if __name__ == "__main__":
    _ = upload_images_only()
