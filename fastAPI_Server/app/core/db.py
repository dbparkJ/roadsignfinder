from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from .config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True) #echo = SQL 로그 출력 여부, future = SQLAlchemy 2.0스타일 사용
"""
engine = "DB와 통신하는 최상위 객체를 의미한다."
커넥션 풀을 관리하고, DB 드라이버(asyncpg)를 사용하며, 직접 쿼리를 날리지 않고 세션이 하도록 한다.

asyncpg 란?
PostgreSQL 전용 비동기 DB 드라이버
asyncpg = Python에서 PostgreSQL에 가장 빠르게 접근하기 위한 비동기 드라이버
SQLAlchemy가 실제 DB 통신을 asyncpg에게 위임
"""
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with SessionLocal() as session:
        yield session

"""
FAST API 의존성 함수로
요청마다 DB 세션을 하나 생성하고, 자동으로 닫아줌
"""