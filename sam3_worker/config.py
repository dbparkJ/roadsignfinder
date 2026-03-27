import os
from pydantic import BaseModel

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://111.111.111.79:8000").rstrip("/")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = _env_bool("MINIO_SECURE", False)
    MINIO_CONNECT_TIMEOUT: float = float(os.getenv("MINIO_CONNECT_TIMEOUT", "2"))
    MINIO_READ_TIMEOUT: float = float(os.getenv("MINIO_READ_TIMEOUT", "5"))
    INFERENCE_SAVE_IMAGES: bool = _env_bool("INFERENCE_SAVE_IMAGES", True)
    SAM3_USE_FP16: bool = _env_bool("SAM3_USE_FP16", True)
    SAM3_LOG_TIMING: bool = _env_bool("SAM3_LOG_TIMING", False)
    SAM3_DEBUG_LOG: bool = _env_bool("SAM3_DEBUG_LOG", False)

    CELERY_BROKER_URL: str = os.getenv("SAM3_CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://111.111.111.216:6379/0"))

    CALLBACK_URL: str = os.getenv("SAM3_CALLBACK_URL", f"{DEFAULT_API_BASE_URL}/pole_type/callback")
    CALLBACK_TOKEN: str = os.getenv("POLE_TYPE_CALLBACK_TOKEN", "change_me")

    TMP_DIR: str = os.getenv("SAM3_TMP_DIR", "/tmp")

    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "/home/geon_lab/doje/final_project/seongbin/roadsignfinder/l_model/weights/best.pt") #수정해야함
    YOLO_CONF: float = float(os.getenv("YOLO_CONF", "0.7"))

    SAM3_MODEL_NAME: str = os.getenv("SAM3_MODEL_NAME", "facebook/sam3")

    V_STRIP_SCALE: float = float(os.getenv("SAM3_V_STRIP_SCALE", "1.0"))
    MIN_MASK_AREA: int = int(os.getenv("SAM3_MIN_MASK_AREA", "400"))


settings = Settings()
