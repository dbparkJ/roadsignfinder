import requests
import random
import string
from concurrent.futures import ThreadPoolExecutor

BASE = "http://111.111.111.164:8000"

def random_string(length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def random_email():
    return f"{random_string(12)}@test.com"

def random_password():
    # 보통 서버 정책 고려해서 숫자+문자 섞음
    return random_string(12)

def register_user(_):
    payload = {
        "email": random_email(),
        "password": random_password(),
        "display_name": f"유저_{random_string(6)}"
    }
    try:
        r = requests.post(f"{BASE}/auth/register", json=payload, timeout=5)
        return r.status_code
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    TOTAL = 1000
    WORKERS = 16  # 동시에 실행할 요청 수

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(register_user, range(TOTAL)))

    print("성공:", results.count(200))
    print("실패:", TOTAL - results.count(200))
