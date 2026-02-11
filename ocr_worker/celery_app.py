from celery import Celery

from .config import settings


def make_celery():
    app = Celery(
        "ocr_worker",
        broker=settings.CELERY_BROKER_URL,
        include=["ocr_worker.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_default_queue="ocr",
    )
    return app


celery_app = make_celery()

__all__ = ["celery_app", "make_celery"]
