import os
from pydantic import BaseModel


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "photos")
    INFERENCE_BUCKET: str = os.getenv("INFERENCE_BUCKET", "inference")
    SAM3_BUCKET: str = os.getenv("SAM3_BUCKET", "sam3")
    INFERENCE_SAVE_IMAGES: bool = os.getenv("INFERENCE_SAVE_IMAGES", "true").lower() == "true"
    SAM3_USE_FP16: bool = os.getenv("SAM3_USE_FP16", "false").lower() == "true"
    SAM3_LOG_TIMING: bool = os.getenv("SAM3_LOG_TIMING", "false").lower() == "true"

    CELERY_BROKER_URL: str = os.getenv("SAM3_CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://111.111.111.216:6379/0"))

    CALLBACK_URL: str = os.getenv("POLE_TYPE_CALLBACK_URL", "http://111.111.111.216:8000/pole_type/callback")
    CALLBACK_TOKEN: str = os.getenv("POLE_TYPE_CALLBACK_TOKEN", "change_me")

    TMP_DIR: str = os.getenv("SAM3_TMP_DIR", "/tmp")

    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "/home/geonws/workspace/2026_project/roadsign_finder/yolo_worker_dev/model/best.pt")
    YOLO_CONF: float = float(os.getenv("YOLO_CONF", "0.8"))

    SAM3_MODEL_NAME: str = os.getenv("SAM3_MODEL_NAME", "facebook/sam3")

    V_STRIP_SCALE: float = float(os.getenv("SAM3_V_STRIP_SCALE", "1.0"))
    H_STRIP_SCALE: float = float(os.getenv("SAM3_H_STRIP_SCALE", "1.0"))
    MIN_MASK_AREA: int = int(os.getenv("SAM3_MIN_MASK_AREA", "400"))


settings = Settings()
