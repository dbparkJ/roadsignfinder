import os
from pydantic import BaseModel

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://111.111.111.79:8000").rstrip("/")


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_CROP_BUCKET: str = os.getenv("MINIO_CROP_BUCKET", "crop")
    INFERENCE_SAVE_IMAGES: bool = os.getenv("INFERENCE_SAVE_IMAGES", "true").lower() == "true"
    INFERENCE_LOG_TIMING: bool = os.getenv("INFERENCE_LOG_TIMING", "false").lower() == "true"
    INFERENCE_DEBUG_LOG: bool = os.getenv("INFERENCE_DEBUG_LOG", "false").lower() == "true"

    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://111.111.111.216:6379/0")

    CALLBACK_URL: str = os.getenv(
        "INFERENCE_CALLBACK_URL",
        f"{DEFAULT_API_BASE_URL}/inference/callback",
    )
    CALLBACK_TOKEN: str = os.getenv("INFERENCE_CALLBACK_TOKEN", "change_me")

    TMP_DIR: str = os.getenv("INFERENCE_TMP_DIR", "/tmp")
    CROP_TMP_DIR: str = os.getenv("CROP_TMP_DIR", "/tmp/inference_crop")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "yolo-seg")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/home/geon_lab/AI_CHOI/2026_roadsign_finder/version2.2/runs/segment/roadsignfinder_ver3.0/l_model/weights/best.pt")


settings = Settings()
