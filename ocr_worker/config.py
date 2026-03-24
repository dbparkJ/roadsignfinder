import os
from pydantic import BaseModel

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://111.111.111.79:8000").rstrip("/")


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_CROP_BUCKET: str = os.getenv("MINIO_CROP_BUCKET", "crop")
    MINIO_OCR_BUCKET: str = os.getenv("MINIO_OCR_BUCKET", "ocr")

    CELERY_BROKER_URL: str = os.getenv("OCR_CELERY_BROKER_URL", os.getenv("CELERY_BROKER_URL", "redis://111.111.111.216:6379/0"))

    CALLBACK_URL: str = os.getenv("OCR_CALLBACK_URL", f"{DEFAULT_API_BASE_URL}/ocr/callback")
    CALLBACK_TOKEN: str = os.getenv("OCR_CALLBACK_TOKEN", "change_me")
    OCR_DEBUG_LOG: bool = os.getenv("OCR_DEBUG_LOG", "false").lower() == "true"
    OCR_RELEASE_GPU_CACHE: bool = os.getenv("OCR_RELEASE_GPU_CACHE", "true").lower() == "true"
    OCR_REUSE_PIPELINE: bool = os.getenv("OCR_REUSE_PIPELINE", "true").lower() == "true"
    OCR_DROP_PIPELINE_AFTER_TASK: bool = os.getenv("OCR_DROP_PIPELINE_AFTER_TASK", "false").lower() == "true"

    TMP_DIR: str = os.getenv("OCR_TMP_DIR", "/tmp/ocr_worker")
    OCR_OUTPUT_DIR: str = os.getenv("OCR_OUTPUT_DIR", "/tmp/paddleocr_output")
    OCR_DEVICE: str = os.getenv("OCR_DEVICE", "gpu:0")
    OCR_USE_QUEUES: bool = os.getenv("OCR_USE_QUEUES", "false").lower() == "true"
    OCR_DISABLE_LAYOUT: bool = os.getenv("OCR_DISABLE_LAYOUT", "true").lower() == "true"
    OCR_DISABLE_ORIENTATION: bool = os.getenv("OCR_DISABLE_ORIENTATION", "true").lower() == "true"
    OCR_DISABLE_UNWARP: bool = os.getenv("OCR_DISABLE_UNWARP", "true").lower() == "true"
    OCR_SMALL_CROP_DOUBLE_EDGE_THRESHOLD: int = int(os.getenv("OCR_SMALL_CROP_DOUBLE_EDGE_THRESHOLD", "200"))
    OCR_SMALL_CROP_DOUBLE_SCALE: float = float(os.getenv("OCR_SMALL_CROP_DOUBLE_SCALE", "2.0"))
    OCR_DEBUG_VARIANTS_ENABLED: bool = os.getenv("OCR_DEBUG_VARIANTS_ENABLED", "false").lower() == "true"
    OCR_DEBUG_VARIANT_COUNT: int = int(os.getenv("OCR_DEBUG_VARIANT_COUNT", "3"))
    OCR_DEBUG_VARIANT_PAD_STEP_PX: int = int(os.getenv("OCR_DEBUG_VARIANT_PAD_STEP_PX", "3"))
    OCR_DEBUG_DIR: str = os.getenv("OCR_DEBUG_DIR", "/tmp/ocr_worker_debug")
    OCR_WORKER_CONCURRENCY: int = int(os.getenv("OCR_WORKER_CONCURRENCY", "1"))
    OCR_WORKER_PREFETCH_MULTIPLIER: int = int(os.getenv("OCR_WORKER_PREFETCH_MULTIPLIER", "1"))
    # 0 or negative means disabled (do not recycle child process by task count).
    OCR_WORKER_MAX_TASKS_PER_CHILD: int = int(os.getenv("OCR_WORKER_MAX_TASKS_PER_CHILD", "0"))


settings = Settings()
