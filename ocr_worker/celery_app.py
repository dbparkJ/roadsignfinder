from celery import Celery

from .config import settings


def make_celery():
    max_tasks_per_child = settings.OCR_WORKER_MAX_TASKS_PER_CHILD
    if max_tasks_per_child <= 0:
        max_tasks_per_child = None

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
        worker_concurrency=settings.OCR_WORKER_CONCURRENCY,
        worker_prefetch_multiplier=settings.OCR_WORKER_PREFETCH_MULTIPLIER,
        worker_max_tasks_per_child=max_tasks_per_child,
    )
    return app


celery_app = make_celery()

__all__ = ["celery_app", "make_celery"]
