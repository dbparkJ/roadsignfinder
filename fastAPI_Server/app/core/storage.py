import os
from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "geonws")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "geonws1234")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "photos")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)
