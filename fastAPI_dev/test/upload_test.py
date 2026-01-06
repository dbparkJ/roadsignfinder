import requests

API = "http://111.111.111.164:8000"
"""
# 1) 로그인해서 JWT 받기
login = requests.post(f"{API}/auth/login", json={
    "email": "pjmsm0319@naver.com",
    "password": "1234"
})
login.raise_for_status()
token = login.json()["access_token"]
print(token)
"""
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzYTI5ZGZmZS0yZTZjLTQ2OGUtOTYzNC1iMWM1YzY3Y2QxZWUiLCJyb2xlIjoidXNlciIsImlhdCI6MTc2NzE2NDA3OCwiZXhwIjoxNzY3MTY1ODc4fQ.3CcsqEjft-Kp2IokZvW3gW-rNXCKentlE07Pf8aXwHY"
headers = {"Authorization": f"Bearer {token}"}

# 2) presign 발급(✅ JWT 필요)
presign = requests.post(f"{API}/storage/presign", json={
    "filename": "sample3.jpg",
    "content_type": "image/jpeg"
}, headers=headers)
presign.raise_for_status()

upload_url = presign.json()["upload_url"]
object_key = presign.json()["object_key"]

# 3) MinIO로 직접 업로드(PUT) (❗여긴 JWT가 아니라 presigned URL로 권한이 있음)
with open(r"D:\RoadSign\sample3.jpg", "rb") as f:
    put = requests.put(upload_url, data=f, headers={"Content-Type": "image/jpeg"})
put.raise_for_status()

# 4) 업로드 완료 등록(✅ JWT 필요)
done = requests.post(f"{API}/photos/complete", json={
    "object_key": object_key,
    "original_filename": "sample.jpg",
    "content_type": "image/jpeg",
}, headers=headers)
done.raise_for_status()

print("OK:", done.json())
