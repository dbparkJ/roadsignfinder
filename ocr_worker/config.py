import os
from pydantic import BaseModel


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_CROP_BUCKET: str = os.getenv("MINIO_CROP_BUCKET", "crop")
    MINIO_OCR_BUCKET: str = os.getenv("MINIO_OCR_BUCKET", "ocr")

    CELERY_BROKER_URL: str = os.getenv("OCR_CELERY_BROKER_URL", os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))

    CALLBACK_URL: str = os.getenv("OCR_CALLBACK_URL", "http://localhost:8000/ocr/callback")
    CALLBACK_TOKEN: str = os.getenv("OCR_CALLBACK_TOKEN", "change_me")

    TMP_DIR: str = os.getenv("OCR_TMP_DIR", "/tmp/ocr_worker")
    OCR_OUTPUT_DIR: str = os.getenv("OCR_OUTPUT_DIR", "/tmp/paddleocr_output")
    OCR_DEVICE: str = os.getenv("OCR_DEVICE", "gpu:0")
    OCR_USE_QUEUES: bool = os.getenv("OCR_USE_QUEUES", "false").lower() == "true"
    OCR_DISABLE_LAYOUT: bool = os.getenv("OCR_DISABLE_LAYOUT", "false").lower() == "true"
    OCR_DISABLE_ORIENTATION: bool = os.getenv("OCR_DISABLE_ORIENTATION", "false").lower() == "true"
    OCR_DISABLE_UNWARP: bool = os.getenv("OCR_DISABLE_UNWARP", "false").lower() == "true"


settings = Settings()
