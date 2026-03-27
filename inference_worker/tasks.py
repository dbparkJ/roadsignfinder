import os
import time
import json
import math
import traceback
import requests
import urllib3
from minio import Minio
from pathlib import Path
from PIL import Image
from shutil import rmtree

from .celery_app import celery_app
from .config import settings
from .inference import run_inference_on_file


def _debug(msg: str) -> None:
    if settings.INFERENCE_DEBUG_LOG:
        print(msg)


def _minio_client():
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
        http_client=urllib3.PoolManager(
            timeout=urllib3.Timeout(
                connect=settings.MINIO_CONNECT_TIMEOUT,
                read=settings.MINIO_READ_TIMEOUT,
            ),
            retries=False,
        ),
    )


def _callback(payload: dict):
    headers = {
        "Content-Type": "application/json",
        "x-inference-token": settings.CALLBACK_TOKEN,
    }
    cleanup_tmp = True
    try:
        resp = requests.post(
            settings.CALLBACK_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=10,
        )
        _debug(
            "[inference] callback_sent "
            f"url={settings.CALLBACK_URL} status={resp.status_code} "
            f"job_id={payload.get('job_id')} state={payload.get('status')}"
        )
        if resp.status_code >= 400:
            body = (resp.text or "").strip().replace("\n", " ")
            if len(body) > 200:
                body = body[:200] + "..."
            print(
                "[WARN] callback non-2xx "
                f"url={settings.CALLBACK_URL} status={resp.status_code} body={body}"
            )
    except Exception as e:
        # 최종 콜백 실패 시 로그만 남김
        print(f"[WARN] callback failed url={settings.CALLBACK_URL}: {e}")


def _safe_token(v: str | None) -> str:
    if not v:
        return "unk"
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(v).strip())
    token = token.strip("_")
    return token or "unk"


def _build_yolo_crops(
    image_path: str,
    boxes: list[dict],
    crop_dir: Path,
    source_stem: str | None = None,
) -> list[dict]:
    crop_dir = crop_dir.expanduser().resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    base_name = _safe_token(source_stem) if source_stem else _safe_token(Path(image_path).stem)

    with Image.open(image_path).convert("RGB") as src:
        width, height = src.size

        for idx, box in enumerate(boxes):
            xyxy = box.get("xyxy") if isinstance(box, dict) else None
            if not (isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4):
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
            except Exception:
                continue

            x1i = max(0, min(width - 1, math.floor(x1)))
            y1i = max(0, min(height - 1, math.floor(y1)))
            x2i = max(1, min(width, math.ceil(x2)))
            y2i = max(1, min(height, math.ceil(y2)))
            if x2i <= x1i or y2i <= y1i:
                continue

            class_id = box.get("class_id") if isinstance(box, dict) else None
            confidence = box.get("confidence") if isinstance(box, dict) else None

            filename = f"{base_name}_crop_{idx:03d}.jpg"
            crop_path = crop_dir / filename

            crop_img = src.crop((x1i, y1i, x2i, y2i))
            crop_img.save(crop_path, format="JPEG", quality=95)

            items.append(
                {
                    "det_index": idx,
                    "crop_path": str(crop_path),
                    "bbox_xyxy": [x1i, y1i, x2i, y2i],
                    "class_id": class_id,
                    "class_name": box.get("class_name") if isinstance(box, dict) else None,
                    "confidence": confidence,
                }
            )

    return items


def _upload_crops_to_minio(
    client: Minio,
    crop_items: list[dict],
    bucket_name: str,
    object_prefix: str,
) -> None:
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    prefix = object_prefix.strip("/")
    for item in crop_items:
        crop_path = item.get("crop_path")
        if not crop_path:
            item["crop_upload"] = "fail"
            item["crop_upload_error"] = "empty crop_path"
            continue

        local_path = Path(str(crop_path)).expanduser().resolve()
        object_key = f"{prefix}/{local_path.name}" if prefix else local_path.name
        try:
            client.fput_object(
                bucket_name,
                object_key,
                str(local_path),
                content_type="image/jpeg",
            )
            item["crop_upload"] = "ok"
            item["crop_bucket"] = bucket_name
            item["crop_object_key"] = object_key
            item["crop_object_uri"] = f"{bucket_name}/{object_key}"
        except Exception as e:
            item["crop_upload"] = "fail"
            item["crop_upload_error"] = str(e)
        finally:
            try:
                if local_path.exists():
                    local_path.unlink()
            except Exception:
                pass


@celery_app.task(name="inference_worker.tasks.run_inference")
def run_inference(
    job_id: str,
    photo_id: str,
    bucket: str,
    object_key: str,
    rdid: str,
    img_x: float,
    img_y: float,
    result_bucket: str,
    result_prefix: str,
):
    _debug(
        "[inference] task_start "
        f"job_id={job_id} callback_url={settings.CALLBACK_URL} "
        f"token_set={'yes' if settings.CALLBACK_TOKEN else 'no'} "
        f"model_path={settings.MODEL_PATH} model_exists={os.path.exists(settings.MODEL_PATH)}"
    )
    client = _minio_client()
    ext = Path(object_key).suffix or ".bin"
    os.makedirs(settings.TMP_DIR, exist_ok=True)
    tmp_file = os.path.join(settings.TMP_DIR, f"{job_id}{ext}")

    # 상태: processing
    _callback({"job_id": job_id, "status": "processing", "result_object_key": None, "result_json": None, "error_message": None})

    cleanup_tmp = True
    try:
        t0 = time.perf_counter()
        try:
            client.stat_object(bucket, object_key)
        except Exception as e:
            raise RuntimeError(f"minio object not found: {bucket}/{object_key} ({e})") from e
        client.fget_object(bucket, object_key, tmp_file)
        if not os.path.exists(tmp_file):
            raise RuntimeError(f"downloaded file missing: {tmp_file}")
        _debug(f"[inference] input_image={tmp_file}")
        t1 = time.perf_counter()
        # 추론 결과 업로드할 버킷 존재 여부 확인
        if not client.bucket_exists(result_bucket):
            client.make_bucket(result_bucket)

        _debug(f"[inference] stage=model_start job_id={job_id}")
        crop_source_path = str(
            Path(settings.CROP_TMP_DIR) / job_id / f"{_safe_token(Path(object_key).stem)}_nafnet.jpg"
        )
        result, annotated_path, crop_source_image_path = run_inference_on_file(
            tmp_file,
            job_id,
            photo_id,
            rdid,
            img_x,
            img_y,
            crop_source_path=crop_source_path,
        )
        _debug(f"[inference] stage=model_done job_id={job_id}")
        no_detections = result.get("no_detections")
        if no_detections is False:
            cleanup_tmp = False
        t2 = time.perf_counter()
        annotated_key = None
        result_size = None
        if not no_detections and settings.INFERENCE_SAVE_IMAGES:
            if not annotated_path or not os.path.exists(annotated_path):
                raise RuntimeError("annotated mask file missing despite detections")

            annotated_object_key = f"{result_prefix}/{photo_id}/inference/{job_id}.jpg"
            try:
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
            finally:
                try:
                    os.remove(annotated_path)
                except Exception:
                    pass
            _debug(f"[inference] output_image={annotated_key}")
        elif not no_detections and not settings.INFERENCE_SAVE_IMAGES:
            _debug("[inference] output_image=skipped")
        t3 = time.perf_counter()

        if not no_detections:
            crops_root = Path(settings.CROP_TMP_DIR) / job_id
            crop_items = _build_yolo_crops(
                crop_source_image_path or tmp_file,
                result.get("boxes") or [],
                crops_root,
                source_stem=Path(object_key).stem,
            )
            _upload_crops_to_minio(
                client=client,
                crop_items=crop_items,
                bucket_name=settings.MINIO_CROP_BUCKET,
                object_prefix=f"{result_prefix}/{photo_id}/crop/{job_id}",
            )
            result["crop_images"] = crop_items
            try:
                if crops_root.exists():
                    rmtree(crops_root)
            except Exception:
                pass
        else:
            result["crop_images"] = []
        t4 = time.perf_counter()

        if settings.INFERENCE_LOG_TIMING:
            print(
                "[inference] timing download={:.3f}s infer={:.3f}s upload={:.3f}s crop={:.3f}s total={:.3f}s".format(
                    t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0
                )
            )

        _callback(
            {
                "job_id": job_id,
                "status": "done",
                "result_object_key": annotated_key,
                "result_json": {**result, "annotated_key": annotated_key} if not no_detections else {"no_detections": True, "selected": "none", "crop_images": []},
                "size_bytes": result_size,
                "error_message": None,
            }
        )
    except Exception as e:
        print(f"[ERROR] run_inference failed job_id={job_id}: {e}")
        print(traceback.format_exc())
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
            if cleanup_tmp and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
