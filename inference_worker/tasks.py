import os
import json
import io
import requests
from datetime import datetime, timezone
from minio import Minio
from pathlib import Path

from .celery_app import celery_app
from .config import settings
from .inference import run_inference_on_file


def _minio_client():
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def _callback(payload: dict):
    headers = {
        "Content-Type": "application/json",
        "x-inference-token": settings.CALLBACK_TOKEN,
    }
    try:
        requests.post(settings.CALLBACK_URL, headers=headers, data=json.dumps(payload), timeout=10)
    except Exception as e:
        # 최종 콜백 실패 시 로그만 남김
        print(f"[WARN] callback failed: {e}")


@celery_app.task(name="inference_worker.tasks.run_inference")
def run_inference(job_id: str, photo_id: str, bucket: str, object_key: str, rdid: str, img_x: float, img_y: float):
    client = _minio_client()
    ext = Path(object_key).suffix or ".bin"
    os.makedirs(settings.TMP_DIR, exist_ok=True)
    tmp_file = os.path.join(settings.TMP_DIR, f"{job_id}{ext}")

    # 상태: processing
    _callback({"job_id": job_id, "status": "processing", "result_object_key": None, "result_json": None, "error_message": None})

    try:
        client.fget_object(bucket, object_key, tmp_file)
        # 추론 결과 업로드할 버킷 존재 여부 확인
        if not client.bucket_exists(settings.INFERENCE_BUCKET):
            client.make_bucket(settings.INFERENCE_BUCKET)

        result, annotated_path = run_inference_on_file(tmp_file, job_id, photo_id, rdid, img_x, img_y, settings.SELECTION_ALPHA)
        no_detections = result.get("no_detections")
        annotated_key = None
        result_size = None
        if not no_detections:
            if not annotated_path or not os.path.exists(annotated_path):
                raise RuntimeError("annotated mask file missing despite detections")

            annotated_object_key = f"{photo_id}/inference/{job_id}.jpg"
            try:
                with open(annotated_path, "rb") as f:
                    size = os.path.getsize(annotated_path)
                    client.put_object(
                        settings.INFERENCE_BUCKET,
                        annotated_object_key,
                        data=f,
                        length=size,
                        content_type="image/jpeg",
                    )
                    annotated_key = f"{settings.INFERENCE_BUCKET}/{annotated_object_key}"
                    result_size = size
            finally:
                try:
                    os.remove(annotated_path)
                except Exception:
                    pass

        _callback(
            {
                "job_id": job_id,
                "status": "done",
                "result_object_key": annotated_key,
                "result_json": {**result, "annotated_key": annotated_key} if not no_detections else {"no_detections": True, "selected": "none"},
                "size_bytes": result_size,
                "error_message": None,
            }
        )
    except Exception as e:
        _callback(
            {
                "job_id": job_id,
                "status": "failed",
                "result_object_key": None,
                "result_json": None,
                "error_message": str(e),
            }
        )
    finally:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
