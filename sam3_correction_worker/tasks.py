import json
import mimetypes
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from minio import Minio
from PIL import Image

from fastAPI_Server.app.utils.inference import bbox_iou, mask_bbox, normalize_bbox, select_nearest_detection
from sam3_worker.sam3_inference import DEVICE, _get_sam3, _get_yolo

from .celery_app import celery_app
from .config import settings


def _debug(msg: str) -> None:
    if settings.SAM3_DEBUG_LOG:
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
        "x-class-correction-token": settings.CALLBACK_TOKEN,
    }
    try:
        requests.post(settings.CALLBACK_URL, headers=headers, data=json.dumps(payload), timeout=10)
    except Exception as e:
        print(f"[WARN] class correction callback failed: {e}")


def _names_to_inv(names) -> dict[str, int]:
    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, list):
        items = enumerate(names)
    else:
        items = []

    inv: dict[str, int] = {}
    for class_id, class_name in items:
        name = str(class_name).strip()
        if not name:
            continue
        inv[name] = int(class_id)
        inv[name.replace(" ", "")] = int(class_id)
    return inv


def _resolve_class_id(class_id, class_name, names_inv: dict[str, int]) -> int | None:
    if class_id is not None:
        try:
            return int(class_id)
        except Exception:
            pass

    if class_name is None:
        return None

    text = str(class_name).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return names_inv.get(text) or names_inv.get(text.replace(" ", ""))


def _fallback_box(img_x: float | None, img_y: float | None, width: int, height: int, half_size: int = 32) -> list[float]:
    cx = float(img_x if img_x is not None else width / 2.0)
    cy = float(img_y if img_y is not None else height / 2.0)
    x1 = max(0.0, cx - half_size)
    y1 = max(0.0, cy - half_size)
    x2 = min(float(width - 1), cx + half_size)
    y2 = min(float(height - 1), cy + half_size)
    return [x1, y1, x2, y2]


def _to_numpy_mask(mask) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        arr = mask
    else:
        arr = np.asarray(mask)
    if arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    return (arr > 0).astype(np.uint8)


def _largest_polygon_from_mask(mask) -> list[list[float]] | None:
    mask_u8 = _to_numpy_mask(mask)
    if mask_u8 is None:
        return None

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 3:
        return None

    polygon: list[list[float]] = []
    for point in contour:
        if len(point) < 1 or len(point[0]) < 2:
            continue
        x, y = point[0]
        polygon.append([float(x), float(y)])
    return polygon if len(polygon) >= 3 else None


def _polygon_from_bbox(box_xyxy) -> list[list[float]] | None:
    bbox = normalize_bbox(box_xyxy)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)],
    ]


def _polygon_for_detection(det: dict) -> list[list[float]] | None:
    mask = det.get("mask")
    if isinstance(mask, list) and len(mask) >= 3:
        return [
            [float(point[0]), float(point[1])]
            for point in mask
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ] or None
    box = det.get("box_xyxy")
    if box is None:
        box = mask_bbox(mask)
    return _polygon_from_bbox(box)


def _normalize_polygon(polygon: list[list[float]], width: int, height: int) -> list[float]:
    width = max(1, width)
    height = max(1, height)
    flattened: list[float] = []
    for x, y in polygon:
        flattened.append(max(0.0, min(1.0, float(x) / width)))
        flattened.append(max(0.0, min(1.0, float(y) / height)))
    return flattened


def _format_yolo_seg_line(class_id: int, polygon: list[list[float]], width: int, height: int) -> str | None:
    normalized = _normalize_polygon(polygon, width, height)
    if len(normalized) < 6:
        return None
    coords = " ".join(f"{value:.6f}" for value in normalized)
    return f"{class_id} {coords}"


def _select_best_result_mask(results: dict, prompt_box: list[float]):
    masks = results.get("masks") or []
    boxes = results.get("boxes") or []
    prompt_bbox = normalize_bbox(prompt_box)

    best_mask = None
    best_score = None
    for idx, mask in enumerate(masks):
        candidate_box = None
        if isinstance(boxes, list) and idx < len(boxes):
            candidate_box = normalize_bbox(boxes[idx])
        if candidate_box is None:
            polygon = _largest_polygon_from_mask(mask)
            if polygon:
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
                candidate_box = (min(xs), min(ys), max(xs), max(ys))
        score = bbox_iou(prompt_bbox, candidate_box) if prompt_bbox and candidate_box else 0.0
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    return best_mask


def _run_sam3_for_box(image: Image.Image, prompt_box: list[float]):
    model, processor = _get_sam3()
    inputs = processor(
        images=image,
        input_boxes=[[prompt_box]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return _select_best_result_mask(results, prompt_box)


@celery_app.task(name="sam3_correction_worker.tasks.run_correction")
def run_correction(
    correction_id: str,
    photo_id: str,
    source_bucket: str,
    source_object_key: str,
    photo_name: str,
    rdid: str,
    class_name: str,
    img_x: float | None,
    img_y: float | None,
    existing_detections: list[dict] | None,
    upload_bucket: str,
    upload_image_object_key: str,
    upload_label_object_key: str,
):
    client = _minio_client()
    tmp_root = Path(settings.TMP_DIR).expanduser().resolve() / correction_id
    tmp_root.mkdir(parents=True, exist_ok=True)
    source_path = tmp_root / (Path(source_object_key).name or f"{correction_id}.jpg")
    label_path = tmp_root / (Path(upload_label_object_key).name or f"{correction_id}.txt")

    _callback(
        {
            "correction_id": correction_id,
            "status": "processing",
            "upload_bucket": upload_bucket,
            "upload_image_object_key": upload_image_object_key,
            "upload_label_object_key": upload_label_object_key,
            "result_json": None,
            "error_message": None,
        }
    )

    try:
        if not client.bucket_exists(upload_bucket):
            client.make_bucket(upload_bucket)

        client.fget_object(source_bucket, source_object_key, str(source_path))
        image_content_type = mimetypes.guess_type(photo_name or str(source_path.name))[0] or "application/octet-stream"
        client.fput_object(
            upload_bucket,
            upload_image_object_key,
            str(source_path),
            content_type=image_content_type,
        )

        with Image.open(source_path).convert("RGB") as image:
            width, height = image.size
            detections = existing_detections or []
            target_det = None
            if img_x is not None and img_y is not None:
                target_det = select_nearest_detection(detections, img_x, img_y)
            prompt_box = normalize_bbox(target_det.get("box_xyxy")) if target_det else None
            if prompt_box is None and target_det:
                prompt_box = mask_bbox(target_det.get("mask"))
            if prompt_box is None:
                prompt_box = _fallback_box(img_x, img_y, width, height)

            target_mask = _run_sam3_for_box(image, list(prompt_box))
            target_polygon = _largest_polygon_from_mask(target_mask) or _polygon_from_bbox(prompt_box)
            if target_polygon is None:
                raise RuntimeError("failed to build target polygon")

            yolo = _get_yolo()
            names_inv = _names_to_inv(getattr(yolo, "names", {}))
            user_class_id = _resolve_class_id(None, class_name, names_inv)
            if user_class_id is None:
                raise RuntimeError(f"user class_name not found in model labels: {class_name}")

            lines: list[str] = []
            target_line = _format_yolo_seg_line(user_class_id, target_polygon, width, height)
            if target_line is None:
                raise RuntimeError("failed to format target yolo-seg line")
            lines.append(target_line)

            selected_index = target_det.get("index") if isinstance(target_det, dict) else None
            preserved = 0
            for det in detections:
                if selected_index is not None and det.get("index") == selected_index:
                    continue
                polygon = _polygon_for_detection(det)
                if polygon is None:
                    continue
                det_class_id = _resolve_class_id(det.get("class_id"), det.get("class_name"), names_inv)
                if det_class_id is None:
                    continue
                line = _format_yolo_seg_line(det_class_id, polygon, width, height)
                if line is None:
                    continue
                lines.append(line)
                preserved += 1

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        client.fput_object(
            upload_bucket,
            upload_label_object_key,
            str(label_path),
            content_type="text/plain",
        )

        _callback(
            {
                "correction_id": correction_id,
                "status": "done",
                "upload_bucket": upload_bucket,
                "upload_image_object_key": upload_image_object_key,
                "upload_label_object_key": upload_label_object_key,
                "result_json": {
                    "photo_id": photo_id,
                    "rdid": rdid,
                    "class_name": class_name,
                    "resolved_class_id": user_class_id,
                    "prompt_box_xyxy": list(prompt_box),
                    "selected_detection_index": selected_index,
                    "annotations_total": len(lines),
                    "preserved_existing_annotations": preserved,
                },
                "error_message": None,
            }
        )
    except Exception as e:
        print(f"[ERROR] class correction worker failed correction_id={correction_id}: {e}")
        print(traceback.format_exc())
        _callback(
            {
                "correction_id": correction_id,
                "status": "failed",
                "upload_bucket": upload_bucket,
                "upload_image_object_key": upload_image_object_key,
                "upload_label_object_key": upload_label_object_key,
                "result_json": None,
                "error_message": str(e),
            }
        )
    finally:
        try:
            if source_path.exists():
                source_path.unlink()
        except Exception:
            pass
        try:
            if label_path.exists():
                label_path.unlink()
        except Exception:
            pass
        try:
            if tmp_root.exists():
                tmp_root.rmdir()
        except Exception:
            pass
