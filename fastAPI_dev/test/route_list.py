import requests

API = "http://111.111.111.164:8000"

openapi = requests.get(f"{API}/openapi.json").json()
paths = sorted(openapi["paths"].keys())

for p in paths:
    if "upload" in p or "minio" in p or "internal" in p or "storage" in p:
        print(p)
