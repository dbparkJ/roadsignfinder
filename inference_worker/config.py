import os
from pydantic import BaseModel


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "photos")  # 원본 이미지 버킷
    INFERENCE_BUCKET: str = os.getenv("INFERENCE_BUCKET", "inference")  # 추론 결과 버킷

    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://111.111.111.216:6379/0")

    CALLBACK_URL: str = os.getenv("INFERENCE_CALLBACK_URL", "http://111.111.111.216:8000/inference/callback")
    CALLBACK_TOKEN: str = os.getenv("INFERENCE_CALLBACK_TOKEN", "change_me")

    TMP_DIR: str = os.getenv("INFERENCE_TMP_DIR", "/tmp")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "yolo-seg")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/home/geonws/workspace/2026_project/roadsign_finder/yolo_worker/model/best.pt")
    SELECTION_ALPHA: float = float(os.getenv("SELECTION_ALPHA", "0.001"))  # conf - alpha*dist 가중치


settings = Settings()
