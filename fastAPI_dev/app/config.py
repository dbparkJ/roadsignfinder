from pydantic import BaseModel
import os

class Settings(BaseModel):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://jm:jm1234@111.111.111.216:5432/jm")

    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_TO_LONG_RANDOM") # 실 서비스시 CHANGE_ME_TO_LONG_RANDOM 부분 바꾸기
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256") #JSON Web Token 알고리즘 중 1개
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

settings = Settings()
