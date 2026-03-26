import os
from pydantic import BaseModel

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://111.111.111.79:8000").rstrip("/")


class Settings(BaseModel):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "geonws")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "geonws1234")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    CELERY_BROKER_URL: str = os.getenv(
        "SAM3_CORRECTION_CELERY_BROKER_URL",
        os.getenv("SAM3_CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://111.111.111.216:6379/0")),
    )
    CALLBACK_URL: str = os.getenv(
        "CLASS_CORRECTION_CALLBACK_URL",
        f"{DEFAULT_API_BASE_URL}/class-corrections/callback",
    )
    CALLBACK_TOKEN: str = os.getenv("CLASS_CORRECTION_CALLBACK_TOKEN", "change_me")

    TMP_DIR: str = os.getenv("SAM3_CORRECTION_TMP_DIR", "/tmp")
    SAM3_DEBUG_LOG: bool = os.getenv("SAM3_CORRECTION_DEBUG_LOG", "false").lower() == "true"


settings = Settings()
