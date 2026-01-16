import json
import os
import requests
import time
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from .celery_app import celery_app
from .config import settings
from .sam3_inference import run_sam3_on_file


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
        "x-pole-type-token": settings.CALLBACK_TOKEN,
    }
    try:
        requests.post(settings.CALLBACK_URL, headers=headers, data=json.dumps(payload), timeout=10)
    except Exception as e:
        print(f"[WARN] sam3 callback failed: {e}")


@celery_app.task(name="sam3_worker.tasks.run_sam3")
def run_sam3(
    job_id: str,
    photo_id: str,
    bucket: str,
    object_key: str,
    rdid: str,
    result_bucket: str,
    result_prefix: str,
):
    client = _minio_client()
    ext = Path(object_key).suffix or ".bin"
    os.makedirs(settings.TMP_DIR, exist_ok=True)
    tmp_file = os.path.join(settings.TMP_DIR, f"{job_id}{ext}")
    overlay_file = os.path.join(settings.TMP_DIR, f"{job_id}_yolo.jpg")
    annotated_path = None

    use_existing = os.path.exists(tmp_file)
    if use_existing:
        print(f"[pole_type] reuse_tmp={tmp_file}")
    else:
        print(f"[pole_type] download bucket={bucket} object_key={object_key} tmp={tmp_file}")
    _callback({"job_id": job_id, "status": "processing", "result_object_key": None, "result_json": None, "error_message": None})

    try:
        t0 = time.perf_counter()
        if not use_existing:
            try:
                client.stat_object(bucket, object_key)
            except S3Error as e:
                raise RuntimeError(f"minio stat failed: {bucket}/{object_key} ({e})") from e
            except Exception as e:
                raise RuntimeError(f"minio object not found: {bucket}/{object_key} ({e})") from e
            client.fget_object(bucket, object_key, tmp_file)
        if not os.path.exists(tmp_file):
            raise RuntimeError(f"downloaded file missing: {tmp_file}")
        t1 = time.perf_counter()
        if settings.INFERENCE_SAVE_IMAGES and not client.bucket_exists(result_bucket):
            client.make_bucket(result_bucket)

        overlay_base = None
        yolo_object_key = f"{result_prefix}/{photo_id}/inference/{job_id}.jpg"
        if settings.INFERENCE_SAVE_IMAGES:
            try:
                client.stat_object(result_bucket, yolo_object_key)
                client.fget_object(result_bucket, yolo_object_key, overlay_file)
                if os.path.exists(overlay_file):
                    overlay_base = overlay_file
            except Exception:
                overlay_base = None

        result, annotated_path = run_sam3_on_file(tmp_file, job_id, photo_id, rdid, overlay_base_path=overlay_base)
        t2 = time.perf_counter()
        annotated_key = None
        result_size = None
        if annotated_path and os.path.exists(annotated_path) and settings.INFERENCE_SAVE_IMAGES:
            annotated_object_key = f"{result_prefix}/{photo_id}/inference/{job_id}.jpg"
            with open(annotated_path, "rb") as f:
                size = os.path.getsize(annotated_path)
                client.put_object(
                    result_bucket,
                    annotated_object_key,
                    data=f,
                    length=size,
                    content_type="image/jpeg",
                )
                annotated_key = f"{result_bucket}/{annotated_object_key}"
                result_size = size
            print(f"[pole_type] output_image={annotated_key}")
        elif annotated_path and not settings.INFERENCE_SAVE_IMAGES:
            print("[pole_type] output_image=skipped")
        t3 = time.perf_counter()
        if settings.SAM3_LOG_TIMING:
            print(
                "[pole_type] timing download={:.3f}s infer={:.3f}s upload={:.3f}s total={:.3f}s".format(
                    t1 - t0, t2 - t1, t3 - t2, t3 - t0
                )
            )

        _callback(
            {
                "job_id": job_id,
                "status": "done",
                "result_object_key": annotated_key,
                "result_json": result,
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
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        try:
            if overlay_file and os.path.exists(overlay_file):
                os.remove(overlay_file)
        except Exception:
            pass
        try:
            if annotated_path and os.path.exists(annotated_path):
                os.remove(annotated_path)
        except Exception:
            pass
