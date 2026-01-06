# worker/celery_app.py
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://111.111.111.216:6379/0")

celery = Celery(
    "yolo_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],  # ✅ tasks 강제 import
)

celery.conf.task_routes = {
    "run_inference": {"queue": "yolo"},
}

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

