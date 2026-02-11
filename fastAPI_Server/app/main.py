# app/main.py
import traceback

from fastapi import FastAPI, HTTPException, Request

from .core.db import Base, engine
from .core.storage import minio_client, MINIO_BUCKET
from .routers import auth, health, photos, inference, pole_type
from .services.upload import log_error

print("### LOADED app.main ###")

app = FastAPI(title="JWT Auth + MinIO + Inference")

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(photos.router)
app.include_router(inference.router)
app.include_router(pole_type.router)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
            print(f"[OK] Created bucket: {MINIO_BUCKET}")
    except Exception as e:
        print(f"[WARN] MinIO bucket check/create failed: {e}")


@app.middleware("http")
async def log_unhandled_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except HTTPException as e:
        await log_error(
            path=str(request.url),
            method=request.method,
            status_code=e.status_code,
            message=str(e.detail),
            stacktrace=None,
        )
        raise
    except Exception as e:
        await log_error(
            path=str(request.url),
            method=request.method,
            status_code=500,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )
        raise
