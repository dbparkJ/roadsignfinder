import os
import time
import requests


API_BASE = os.getenv("API_BASE", "http://111.111.111.216:8000")
EMAIL = os.getenv("API_EMAIL", "pjmsm0319@naver.com")
PASSWORD = os.getenv("API_PASSWORD", "1234")
FILE_PATH = os.getenv("UPLOAD_FILE", "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK03/Camera01/Job_20250731_1053_Track13_Sphere_00172.jpg")

IMG_X = float(os.getenv("IMG_X", "100"))
IMG_Y = float(os.getenv("IMG_Y", "100"))
GEO_X = float(os.getenv("GEO_X", "337917.591"))
GEO_Y = float(os.getenv("GEO_Y", "4190091.317"))
RDID = os.getenv("RDID", "R23150641650000100020250187")

POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))
TIMEOUT_SEC = float(os.getenv("TIMEOUT_SEC", "120"))


def login_headers() -> dict:
    r = requests.post(
        f"{API_BASE}/auth/login",
        json={"email": EMAIL, "password": PASSWORD},
        timeout=20,
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload_photo(headers: dict) -> dict:
    if not FILE_PATH:
        raise RuntimeError("UPLOAD_FILE env var is required.")
    with open(FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(FILE_PATH), f, "image/jpeg")}
        data = {
            "img_x": IMG_X,
            "img_y": IMG_Y,
            "geo_x": GEO_X,
            "geo_y": GEO_Y,
            "rdid": RDID,
        }
        r = requests.post(f"{API_BASE}/photos", headers=headers, files=files, data=data, timeout=60)
        if not r.ok:
            raise RuntimeError(f"upload failed: status={r.status_code} body={r.text}")
        return r.json()


def fetch_inference(headers: dict, photo_id: str) -> dict | None:
    r = requests.get(f"{API_BASE}/inference/result", headers=headers, params={"photo_id": photo_id}, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_pole_type(headers: dict, photo_id: str) -> dict | None:
    r = requests.get(f"{API_BASE}/pole_type/result", headers=headers, params={"photo_id": photo_id}, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def main():
    headers = login_headers()
    photo = upload_photo(headers)
    photo_id = photo["id"]
    print(f"[upload] photo_id={photo_id}")

    deadline = time.time() + TIMEOUT_SEC
    last_inf = None
    last_pole = None
    while time.time() < deadline:
        inf = fetch_inference(headers, photo_id)
        pole = fetch_pole_type(headers, photo_id)

        if inf and inf != last_inf:
            print(f"[inference] status={inf.get('status')} result_key={inf.get('result_object_key')}")
            last_inf = inf
        if pole and pole != last_pole:
            print(f"[pole_type] status={pole.get('status')} result_key={pole.get('result_object_key')}")
            last_pole = pole

        inf_done = inf and inf.get("status") in ("done", "failed")
        pole_done = pole and pole.get("status") in ("done", "failed")
        if inf_done and pole_done:
            print("[done] inference and pole_type finished.")
            return

        time.sleep(POLL_SEC)

    raise RuntimeError("timeout waiting for inference/pole_type")


if __name__ == "__main__":
    main()
