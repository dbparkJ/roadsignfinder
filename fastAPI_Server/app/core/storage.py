import os

import urllib3
from minio import Minio


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "125.142.22.24:56000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "geonws")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "geonws1234")
MINIO_SECURE = _env_bool("MINIO_SECURE", False)
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "photos")
MINIO_CONNECT_TIMEOUT = float(os.getenv("MINIO_CONNECT_TIMEOUT", "2"))
MINIO_READ_TIMEOUT = float(os.getenv("MINIO_READ_TIMEOUT", "5"))

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
    http_client=urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=MINIO_CONNECT_TIMEOUT,
            read=MINIO_READ_TIMEOUT,
        ),
        retries=False,
    ),
)
