# yolo_worker/pred_render_store.py
from __future__ import annotations
import os
import uuid
from typing import Sequence
from PIL import Image

from .yolo_overlay import Detection, draw_yolo_style_overlay

def render_and_upload_pred(
    *,
    minio,
    src_img: Image.Image,
    detections: Sequence[Detection],
    pred_bucket: str,
    object_key: str,
    tmp_dir: str = "/tmp",
    jpeg_quality: int = 90,
) -> str:
    """
    (CPU) detections로 오버레이 이미지 생성 후 MinIO 업로드.
    """
    rendered = draw_yolo_style_overlay(src_img, detections)

    tmp_path = os.path.join(tmp_dir, f"pred_{uuid.uuid4().hex}.jpg")
    rendered.save(tmp_path, format="JPEG", quality=jpeg_quality)

    try:
        minio.fput_object(pred_bucket, object_key, tmp_path, content_type="image/jpeg")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return object_key

