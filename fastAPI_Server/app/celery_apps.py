from celery import Celery

from .core.config import settings

celery_app = Celery(
    "inference_worker",
    broker=settings.CELERY_BROKER_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="inference",
)

sam3_celery_app = Celery(
    "sam3_worker",
    broker=settings.SAM3_CELERY_BROKER_URL,
)
sam3_celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="sam3",
)
