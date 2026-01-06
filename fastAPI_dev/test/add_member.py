import requests

BASE = "http://111.111.111.164:8000"
r = requests.post(f"{BASE}/auth/register", json={
    "email": "pjmsm0319@naver.com",
    "password": "1234",
    "display_name": "종민"
})

print(r.status_code, r.text)
