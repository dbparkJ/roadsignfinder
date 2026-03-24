import json
import time
import shutil
import traceback
from pathlib import Path

import requests
from minio import Minio

from fastAPI_Server.app.ocr_policy import build_ocr_queue_items
from .paddle_ocr import run_ocr_on_crops, release_ocr_runtime
from .preprocess import prepare_ocr_crops

from .celery_app import celery_app
from .config import settings


def _debug(msg: str) -> None:
    if settings.OCR_DEBUG_LOG:
        print(msg)


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
        resp = requests.post(
            settings.CALLBACK_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=10,
        )
        _debug(
            "[ocr] callback_sent "
            f"url={settings.CALLBACK_URL} status={resp.status_code} "
            f"job_id={payload.get('job_id')} state={payload.get('status')}"
        )
        if resp.status_code >= 400:
            body = (resp.text or "").strip().replace("\n", " ")
            if len(body) > 200:
                body = body[:200] + "..."
            print(
                "[WARN] ocr callback non-2xx "
                f"url={settings.CALLBACK_URL} status={resp.status_code} body={body}"
            )
    except Exception as e:
        print(f"[WARN] ocr callback failed url={settings.CALLBACK_URL}: {e}")


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


def _variant_score(item: dict) -> tuple[int, int, int, int]:
    texts = item.get("detected_texts") or []
    char_count = sum(len(str(text).replace("\n", "").strip()) for text in texts)
    padding_px = int(item.get("ocr_debug_padding_px") or 0)
    num_results = int(item.get("num_results") or 0)
    return (len(texts), char_count, num_results, -padding_px)


def _collapse_debug_variants(ocr_payload: dict) -> dict:
    groups: dict[str, list[dict]] = {}
    for item in ocr_payload.get("items", []):
        key = str(item.get("ocr_debug_group") or f"det_{item.get('det_index')}")
        groups.setdefault(key, []).append(item)

    collapsed_items: list[dict] = []
    for key, variants in groups.items():
        if len(variants) == 1:
            collapsed_items.append(variants[0])
            continue

        ordered_variants = sorted(
            variants,
            key=lambda item: int(item.get("ocr_debug_padding_px") or 0),
        )
        best = max(variants, key=_variant_score)
        merged = dict(best)
        merged["debug_variants"] = [
            {
                "variant": item.get("ocr_debug_variant"),
                "padding_px": item.get("ocr_debug_padding_px"),
                "status": item.get("status"),
                "num_results": item.get("num_results"),
                "detected_texts": item.get("detected_texts") or [],
                "crop_path": item.get("crop_path"),
            }
            for item in ordered_variants
        ]
        merged["ocr_debug_group"] = key
        merged["ocr_debug_selected_variant"] = best.get("ocr_debug_variant")
        collapsed_items.append(merged)

    ocr_payload["items"] = collapsed_items
    ocr_payload["total"] = len(collapsed_items)
    ocr_payload["ok"] = sum(1 for item in collapsed_items if item.get("status") == "ok")
    ocr_payload["fail"] = sum(1 for item in collapsed_items if item.get("status") != "ok")
    if ocr_payload["fail"] == 0:
        ocr_payload["status"] = "ok"
    elif ocr_payload["ok"] == 0:
        ocr_payload["status"] = "fail"
    else:
        ocr_payload["status"] = "partial"
    return ocr_payload


def _write_debug_summary(debug_dir: Path, ocr_payload: dict) -> None:
    summary = {
        "status": ocr_payload.get("status"),
        "total": ocr_payload.get("total"),
        "ok": ocr_payload.get("ok"),
        "fail": ocr_payload.get("fail"),
        "items": [],
    }
    for item in ocr_payload.get("items", []):
        summary["items"].append(
            {
                "det_index": item.get("det_index"),
                "class_id": item.get("class_id"),
                "class_name": item.get("class_name"),
                "bbox_xyxy": item.get("bbox_xyxy"),
                "ocr_preprocess": item.get("ocr_preprocess"),
                "selected_variant": item.get("ocr_debug_selected_variant") or item.get("ocr_debug_variant"),
                "detected_texts": item.get("detected_texts") or [],
                "debug_variants": item.get("debug_variants") or [],
            }
        )
    debug_dir.mkdir(parents=True, exist_ok=True)
    summary_path = debug_dir / "ocr_debug_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


@celery_app.task(name="ocr_worker.tasks.run_ocr")
def run_ocr(
    job_id: str,
    photo_id: str,
    rdid: str,
    crop_items: list[dict] | None = None,
    result_prefix: str = "",
):
    all_crops = build_ocr_queue_items(crop_items or [])
    crops = [item for item in all_crops if item.get("ocr_target")]
    _debug(
        "[ocr] task_start "
        f"job_id={job_id} callback_url={settings.CALLBACK_URL} "
        f"token_set={'yes' if settings.CALLBACK_TOKEN else 'no'} "
        f"crops={len(crop_items or [])} selected={len(crops)} device={settings.OCR_DEVICE}"
    )
    client = _minio_client()
    tmp_root = Path(settings.TMP_DIR).expanduser().resolve() / job_id
    tmp_crops_dir = tmp_root / "crops"
    tmp_prepared_dir = tmp_root / "prepared"
    output_dir = Path(settings.OCR_OUTPUT_DIR).expanduser().resolve() / job_id
    debug_dir = Path(settings.OCR_DEBUG_DIR).expanduser().resolve() / job_id
    tmp_crops_dir.mkdir(parents=True, exist_ok=True)
    tmp_prepared_dir.mkdir(parents=True, exist_ok=True)
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
        if not crops:
            payload = {
                "status": "skipped",
                "engine": "PaddleOCRVL",
                "rdid": rdid,
                "output_dir": str(output_dir),
                "total": len(all_crops),
                "selected": 0,
                "ok": 0,
                "fail": 0,
                "items": [],
                "reason": "no_ocr_target_crops",
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

        t0 = time.perf_counter()
        downloaded, download_failures = _download_crops(client, crops, tmp_crops_dir)
        t1 = time.perf_counter()
        _debug(
            "[ocr] stage=download_done "
            f"ok={len(downloaded)} fail={len(download_failures)} sec={t1 - t0:.3f}"
        )
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

        prepared_crops = prepare_ocr_crops(
            downloaded,
            tmp_prepared_dir,
            debug_enabled=settings.OCR_DEBUG_VARIANTS_ENABLED,
            debug_variant_count=settings.OCR_DEBUG_VARIANT_COUNT,
            debug_pad_step_px=settings.OCR_DEBUG_VARIANT_PAD_STEP_PX,
            debug_dir=debug_dir if settings.OCR_DEBUG_VARIANTS_ENABLED else None,
        )
        _debug(f"[ocr] stage=ocr_start items={len(prepared_crops)}")
        ocr_payload = run_ocr_on_crops(
            crops=prepared_crops,
            output_dir=output_dir,
            device=settings.OCR_DEVICE,
            use_queues=settings.OCR_USE_QUEUES,
            disable_layout=settings.OCR_DISABLE_LAYOUT,
            disable_orientation=settings.OCR_DISABLE_ORIENTATION,
            disable_unwarp=settings.OCR_DISABLE_UNWARP,
        )
        t2 = time.perf_counter()
        _debug(
            "[ocr] stage=ocr_done "
            f"status={ocr_payload.get('status')} ok={ocr_payload.get('ok')} "
            f"fail={ocr_payload.get('fail')} sec={t2 - t1:.3f}"
        )
        if settings.OCR_DEBUG_VARIANTS_ENABLED:
            ocr_payload = _collapse_debug_variants(ocr_payload)
        ocr_payload["rdid"] = rdid
        if download_failures:
            ocr_payload["items"].extend(download_failures)
            ocr_payload["total"] = len(crops)
            ocr_payload["fail"] = int(ocr_payload.get("fail", 0)) + len(download_failures)
            if ocr_payload.get("ok", 0) == 0 and ocr_payload.get("fail", 0) > 0:
                ocr_payload["status"] = "fail"
            elif ocr_payload.get("fail", 0) > 0:
                ocr_payload["status"] = "partial"
        if settings.OCR_DEBUG_VARIANTS_ENABLED:
            _write_debug_summary(debug_dir, ocr_payload)

        uploaded_size = _upload_ocr_json(
            client=client,
            ocr_payload=ocr_payload,
            result_prefix=result_prefix,
            photo_id=photo_id,
            job_id=job_id,
        )
        t3 = time.perf_counter()
        _debug(
            "[ocr] stage=upload_done "
            f"uploaded_size={uploaded_size} sec={t3 - t2:.3f}"
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
        print(f"[ERROR] run_ocr failed job_id={job_id}: {e}")
        print(traceback.format_exc())
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
        if settings.OCR_RELEASE_GPU_CACHE:
            release_ocr_runtime(drop_pipeline=settings.OCR_DROP_PIPELINE_AFTER_TASK)
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
