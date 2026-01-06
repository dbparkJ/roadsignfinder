import os
from celery import Celery

# Redis 서버 주소 (환경변수 우선)
REDIS_HOST = os.getenv("REDIS_HOST", "111.111.111.216")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
BACKEND_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"

celery_client = Celery(
    "api",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

celery_client.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=False,
)
