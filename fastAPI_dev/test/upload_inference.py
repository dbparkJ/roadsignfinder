import os
import time
import mimetypes
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://111.111.111.164:8000"
EMAIL = "pjmsm0319@naver.com"
PASSWORD = "1234"

INPUT_DIR = r"D:\1.표지판 인식\3.학습자료\test"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

UPLOAD_WORKERS = 8       # ✅ 업로드 병렬 수 (4~16 추천)
JOB_WAIT_SEC = 180       # upload_id -> job_id 대기
POLL_SEC = 0.5           # job 상태 폴링 주기

# ---------------- API helpers ----------------
def login_headers():
    r = requests.post(f"{API}/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=20)
    r.raise_for_status()
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def guess_content_type(path: str) -> str:
    ct, _ = mimetypes.guess_type(path)
    return ct or "application/octet-stream"

def iter_images(input_dir: str):
    for root, _, files in os.walk(input_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ALLOWED_EXT:
                yield os.path.join(root, fn)

def presign(headers, filename: str, content_type: str):
    r = requests.post(
        f"{API}/storage/presign",
        json={"filename": filename, "content_type": content_type},
        headers=headers,
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    return j["upload_id"], j["upload_url"]

def upload_file(upload_url: str, filepath: str, content_type: str):
    with open(filepath, "rb") as f:
        r = requests.put(upload_url, data=f, headers={"Content-Type": content_type}, timeout=300)
    r.raise_for_status()

def wait_job(headers, upload_id: str, timeout_sec: int):
    for _ in range(timeout_sec):
        s = requests.get(f"{API}/uploads/{upload_id}", headers=headers, timeout=10)
        s.raise_for_status()
        j = s.json()
        if j.get("job_id"):
            return j["job_id"], j["photo_id"]
        time.sleep(1)
    raise TimeoutError(f"job_id not assigned within {timeout_sec}s (upload_id={upload_id})")

def get_job(headers, job_id: str):
    r = requests.get(f"{API}/inference/jobs/{job_id}", headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_segmentations(headers, photo_id: str):
    r = requests.get(f"{API}/photos/{photo_id}/segmentations", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------- Worker task (thread) ----------------
def enqueue_one(headers, filepath: str):
    """
    presign -> PUT -> wait job_id 까지 해서 큐에 확실히 넣고 job_id를 반환.
    (requests는 스레드에서 써도 OK)
    """
    filename = os.path.basename(filepath)
    ct = guess_content_type(filepath)

    upload_id, upload_url = presign(headers, filename, ct)
    upload_file(upload_url, filepath, ct)
    job_id, photo_id = wait_job(headers, upload_id, JOB_WAIT_SEC)

    return {
        "job_id": job_id,
        "photo_id": photo_id,
        "filepath": filepath,
        "filename": filename,
    }

# ---------------- Main ----------------
def main():
    headers = login_headers()
    paths = list(iter_images(INPUT_DIR))
    if not paths:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    # 1) ✅ 멀티스레드 업로드(큐 삽입)
    jobs = {}  # job_id -> info

    upbar = tqdm(total=len(paths), desc=f"Uploading/enqueue (x{UPLOAD_WORKERS})", dynamic_ncols=True)

    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as ex:
        futures = {ex.submit(enqueue_one, headers, p): p for p in paths}

        for fut in as_completed(futures):
            src = futures[fut]
            try:
                info = fut.result()
                jid = info["job_id"]
                jobs[jid] = {
                    **info,
                    "status": "queued",
                    "progress": 0.0,
                    "done": False,
                    "error": None,
                }
                tqdm.write(f"✅ enqueued: {info['filename']} -> job={jid}")
            except Exception as e:
                tqdm.write(f"❌ enqueue failed: {os.path.basename(src)} -> {e}")
            finally:
                upbar.update(1)

    upbar.close()

    if not jobs:
        print("❌ No jobs were enqueued.")
        return

    total_jobs = len(jobs)

    # 2) ✅ 전체 job 완료 대기 + 결과 수집
    overall = tqdm(total=total_jobs, desc="Overall done", dynamic_ncols=True)
    avgbar = tqdm(total=100, desc="Avg progress (alive)", dynamic_ncols=True, leave=False)

    completed = 0
    success = 0
    failed = 0

    try:
        while completed < total_jobs:
            alive = [jid for jid, info in jobs.items() if not info["done"]]
            progresses = []
            q_cnt = r_cnt = 0

            for jid in alive:
                info = jobs[jid]
                try:
                    job = get_job(headers, jid)
                    st = job.get("status", "unknown")
                    pr = job.get("progress")
                    if pr is None:
                        pr = info["progress"]
                    else:
                        pr = float(pr)

                    pr = max(0.0, min(100.0, pr))
                    info["status"] = st
                    info["progress"] = pr
                    progresses.append(pr)

                    if st == "queued":
                        q_cnt += 1
                    elif st == "running":
                        r_cnt += 1

                    if st in ("succeeded", "failed"):
                        info["done"] = True
                        info["error"] = job.get("error")

                        completed += 1
                        overall.update(1)

                        if st == "failed":
                            failed += 1
                            tqdm.write(f"❌ failed: {info['filename']} ({jid}) err={info['error']}")
                        else:
                            success += 1
                            seg = fetch_segmentations(headers, info["photo_id"])
                            tqdm.write(f"✅ done: {info['filename']} ({jid}) seg={len(seg)}")

                except Exception as e:
                    tqdm.write(f"⚠️ poll error: {info['filename']} ({jid}) -> {e}")

            # 평균 진행률 표시(척도)
            if alive:
                avg = sum(progresses) / max(1, len(progresses)) if progresses else 0.0
                avgbar.n = 0
                avgbar.refresh()
                avgbar.update(avg)

                overall.set_postfix_str(
                    f"alive={len(alive)} queued={q_cnt} running={r_cnt} ok={success} fail={failed}"
                )
            else:
                avgbar.n = 0
                avgbar.refresh()
                avgbar.update(100)

            time.sleep(POLL_SEC)

    finally:
        avgbar.close()
        overall.close()

    print("\n========== SUMMARY ==========")
    print("Total jobs:", total_jobs)
    print("✅ success:", success)
    print("❌ failed :", failed)

if __name__ == "__main__":
    main()
