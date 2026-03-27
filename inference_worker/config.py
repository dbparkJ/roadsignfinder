import os
from pydantic import BaseModel

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://111.111.111.79:8000").rstrip("/")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "125.142.22.24:56000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = _env_bool("MINIO_SECURE", False)
    MINIO_CROP_BUCKET: str = os.getenv("MINIO_CROP_BUCKET", "crop")
    MINIO_CONNECT_TIMEOUT: float = float(os.getenv("MINIO_CONNECT_TIMEOUT", "2"))
    MINIO_READ_TIMEOUT: float = float(os.getenv("MINIO_READ_TIMEOUT", "5"))
    INFERENCE_SAVE_IMAGES: bool = _env_bool("INFERENCE_SAVE_IMAGES", True)
    INFERENCE_LOG_TIMING: bool = _env_bool("INFERENCE_LOG_TIMING", False)
    INFERENCE_DEBUG_LOG: bool = _env_bool("INFERENCE_DEBUG_LOG", False)

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
