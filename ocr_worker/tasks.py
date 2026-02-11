import json
import shutil
from pathlib import Path

import requests
from minio import Minio

from .paddle_ocr import run_ocr_on_crops

from .celery_app import celery_app
from .config import settings


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
        "x-ocr-token": settings.CALLBACK_TOKEN,
    }
    try:
        requests.post(settings.CALLBACK_URL, headers=headers, data=json.dumps(payload), timeout=10)
    except Exception as e:
        print(f"[WARN] ocr callback failed: {e}")


def _download_crops(client: Minio, crop_items: list[dict], tmp_dir: Path) -> tuple[list[dict], list[dict]]:
    ok_items: list[dict] = []
    fail_items: list[dict] = []

    for idx, item in enumerate(crop_items):
        crop_bucket = item.get("crop_bucket") or settings.MINIO_CROP_BUCKET
        crop_object_key = item.get("crop_object_key")
        if not crop_object_key:
            failed = dict(item)
            failed["status"] = "fail"
            failed["error"] = "missing crop_object_key"
            fail_items.append(failed)
            continue

        filename = Path(crop_object_key).name
        local_path = tmp_dir / f"{idx:04d}_{filename}"

        try:
            client.fget_object(crop_bucket, crop_object_key, str(local_path))
            loaded = dict(item)
            loaded["crop_path"] = str(local_path)
            ok_items.append(loaded)
        except Exception as e:
            failed = dict(item)
            failed["status"] = "fail"
            failed["error"] = str(e)
            fail_items.append(failed)

    return ok_items, fail_items


def _upload_ocr_json(
    client: Minio,
    ocr_payload: dict,
    result_prefix: str,
    photo_id: str,
    job_id: str,
) -> int:
    bucket = settings.MINIO_OCR_BUCKET
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    output_dir = Path(str(ocr_payload.get("output_dir", ""))).expanduser().resolve()
    uploaded_size = 0
    prefix = result_prefix.strip("/")
    base_prefix = f"{prefix}/{photo_id}/ocr/{job_id}" if prefix else f"{photo_id}/ocr/{job_id}"

    for item in ocr_payload.get("items", []):
        for page in item.get("pages", []):
            object_keys: list[str] = []
            object_uris: list[str] = []
            upload_errors: list[str] = []
            local_files = list(page.get("json_files", []))

            for path_str in local_files:
                path = Path(path_str).expanduser().resolve()
                if not path.exists():
                    upload_errors.append(f"missing file: {path}")
                    continue
                try:
                    rel = path.relative_to(output_dir).as_posix()
                    object_key = f"{base_prefix}/{rel}"
                except Exception:
                    object_key = f"{base_prefix}/{path.name}"

                try:
                    client.fput_object(
                        bucket,
                        object_key,
                        str(path),
                        content_type="application/json",
                    )
                    object_keys.append(object_key)
                    object_uris.append(f"{bucket}/{object_key}")
                    uploaded_size += path.stat().st_size
                except Exception as e:
                    upload_errors.append(f"{path.name}: {e}")

            page["json_bucket"] = bucket
            page["json_local_files"] = local_files
            page["json_object_keys"] = object_keys
            page["json_object_uris"] = object_uris
            page["json_files"] = object_uris
            if upload_errors:
                page["json_upload_errors"] = upload_errors

    return uploaded_size


@celery_app.task(name="ocr_worker.tasks.run_ocr")
def run_ocr(
    job_id: str,
    photo_id: str,
    rdid: str,
    crop_items: list[dict] | None = None,
    result_prefix: str = "",
):
    client = _minio_client()
    crops = crop_items or []
    tmp_root = Path(settings.TMP_DIR).expanduser().resolve() / job_id
    tmp_crops_dir = tmp_root / "crops"
    output_dir = Path(settings.OCR_OUTPUT_DIR).expanduser().resolve() / job_id
    tmp_crops_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _callback(
        {
            "job_id": job_id,
            "status": "processing",
            "result_json": None,
            "error_message": None,
            "size_bytes": None,
        }
    )

    try:
        downloaded, download_failures = _download_crops(client, crops, tmp_crops_dir)
        if not downloaded:
            payload = {
                "status": "done",
                "engine": "PaddleOCRVL",
                "rdid": rdid,
                "output_dir": str(output_dir),
                "total": len(crops),
                "ok": 0,
                "fail": len(download_failures),
                "items": download_failures,
                "reason": "no_downloaded_crops",
            }
            _callback(
                {
                    "job_id": job_id,
                    "status": "done",
                    "result_json": payload,
                    "error_message": None,
                    "size_bytes": 0,
                }
            )
            return

        ocr_payload = run_ocr_on_crops(
            crops=downloaded,
            output_dir=output_dir,
            device=settings.OCR_DEVICE,
            use_queues=settings.OCR_USE_QUEUES,
            disable_layout=settings.OCR_DISABLE_LAYOUT,
            disable_orientation=settings.OCR_DISABLE_ORIENTATION,
            disable_unwarp=settings.OCR_DISABLE_UNWARP,
        )
        ocr_payload["rdid"] = rdid
        if download_failures:
            ocr_payload["items"].extend(download_failures)
            ocr_payload["total"] = len(crops)
            ocr_payload["fail"] = int(ocr_payload.get("fail", 0)) + len(download_failures)
            if ocr_payload.get("ok", 0) == 0 and ocr_payload.get("fail", 0) > 0:
                ocr_payload["status"] = "fail"
            elif ocr_payload.get("fail", 0) > 0:
                ocr_payload["status"] = "partial"

        uploaded_size = _upload_ocr_json(
            client=client,
            ocr_payload=ocr_payload,
            result_prefix=result_prefix,
            photo_id=photo_id,
            job_id=job_id,
        )
        _callback(
            {
                "job_id": job_id,
                "status": "done",
                "result_json": ocr_payload,
                "error_message": None,
                "size_bytes": uploaded_size,
            }
        )
    except Exception as e:
        _callback(
            {
                "job_id": job_id,
                "status": "failed",
                "result_json": None,
                "error_message": str(e),
                "size_bytes": None,
            }
        )
    finally:
        try:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        except Exception:
            pass
        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
        except Exception:
            pass
