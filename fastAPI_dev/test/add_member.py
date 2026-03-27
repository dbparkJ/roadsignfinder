import requests

BASE = "http://111.111.111.79:8000"
r = requests.post(f"{BASE}/auth/register", json={
    "email": "dkgus0622@geonspace.com",
    "password": "1234",
    "display_name": "시아현 대리"
})

print(r.status_code, r.text)
