from pydantic import BaseModel
import os

class Settings(BaseModel):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/roadsignfinder")

    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "geon") # 실 서비스시 CHANGE_ME_TO_LONG_RANDOM 부분 바꾸기
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256") #JSON Web Token 알고리즘 중 1개
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    INFERENCE_CALLBACK_TOKEN: str = os.getenv("INFERENCE_CALLBACK_TOKEN", "change_me")

settings = Settings()
