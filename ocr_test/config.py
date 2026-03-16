import os

from pydantic import BaseModel


class Settings(BaseModel):
    OCR_DEBUG_LOG: bool = os.getenv("OCR_DEBUG_LOG", "false").lower() == "true"
    OCR_RELEASE_GPU_CACHE: bool = os.getenv("OCR_RELEASE_GPU_CACHE", "true").lower() == "true"
    OCR_REUSE_PIPELINE: bool = os.getenv("OCR_REUSE_PIPELINE", "true").lower() == "true"
    OCR_DROP_PIPELINE_AFTER_TASK: bool = os.getenv("OCR_DROP_PIPELINE_AFTER_TASK", "false").lower() == "true"
    OCR_DEVICE: str = os.getenv("OCR_DEVICE", "gpu:0")
    OCR_USE_QUEUES: bool = os.getenv("OCR_USE_QUEUES", "false").lower() == "true"
    OCR_DISABLE_LAYOUT: bool = os.getenv("OCR_DISABLE_LAYOUT", "false").lower() == "true"
    OCR_DISABLE_ORIENTATION: bool = os.getenv("OCR_DISABLE_ORIENTATION", "false").lower() == "true"
    OCR_DISABLE_UNWARP: bool = os.getenv("OCR_DISABLE_UNWARP", "false").lower() == "true"


settings = Settings()
