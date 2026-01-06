import requests

BASE = "http://111.111.111.216:8000"

login = requests.post(f"{BASE}/auth/login", json={
    "email": "pjmsm0319@naver.com",
    "password": "1234"
}, timeout=10)

login.raise_for_status()
token = login.json()["access_token"]
print("token:", token)

me = requests.get(
    f"{BASE}/auth/me",
    headers={"Authorization": f"Bearer {token}"},
    timeout=10
)

print("me : ", me.json())
print("me status:", me.status_code)
print("me body:", me.text)
