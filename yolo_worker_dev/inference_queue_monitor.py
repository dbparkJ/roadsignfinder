import time
import requests
from tqdm import tqdm

API = "http://111.111.111.164:8000"
EMAIL = "pjmsm0319@naver.com"
PASSWORD = "1234"

POLL_SEC = 0.3
# ✅ queued/running 뿐 아니라 완료/실패도 같이 가져와서 '사라짐'을 잡는다
JOBS_ENDPOINT = f"{API}/inference/jobs?status=queued,running,succeeded,failed&limit=500"

def login_headers():
    r = requests.post(f"{API}/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=20)
    r.raise_for_status()
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def fetch_jobs(headers):
    r = requests.get(JOBS_ENDPOINT, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()  # list[dict]

def main():
    headers = login_headers()

    # job_id -> last_status
    last = {}

    bar = tqdm(total=None, desc="Inference Queue Monitor", unit="poll", dynamic_ncols=True)
    try:
        while True:
            items = fetch_jobs(headers)

            # 현 시점 스냅샷
            cur = {}
            counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
            for x in items:
                jid = x.get("job_id") or x.get("id")
                if not jid:
                    continue
                st = x.get("status", "unknown")
                cur[str(jid)] = st
                if st in counts:
                    counts[st] += 1

            # ✅ 새로 관측된 job
            new_jobs = set(cur.keys()) - set(last.keys())
            for jid in sorted(new_jobs):
                tqdm.write(f"[NEW] {jid} status={cur[jid]}")

            # ✅ 상태 전이 감지
            for jid, st in cur.items():
                prev = last.get(jid)
                if prev is not None and prev != st:
                    tqdm.write(f"[MOVE] {jid} {prev} -> {st}")

            # ✅ (선택) 사라진 job 감지
            # limit 때문에 오래된 job이 밀려나면 "사라짐"이 생길 수 있음
            disappeared = set(last.keys()) - set(cur.keys())
            for jid in sorted(disappeared):
                tqdm.write(f"[DROP] {jid} (not in list; maybe older than limit) last={last[jid]}")

            bar.set_postfix_str(
                f"queued={counts['queued']} running={counts['running']} "
                f"succeeded={counts['succeeded']} failed={counts['failed']} tracked={len(cur)}"
            )

            last = cur
            bar.update(1)
            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        tqdm.write("\nStopped.")
    finally:
        bar.close()

if __name__ == "__main__":
    main()

