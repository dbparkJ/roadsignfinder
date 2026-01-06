# yolo_worker/crop_store.py
import os
import math
import uuid
from typing import Optional

from PIL import Image


def ensure_bucket(minio, bucket: str):
    """버킷 없으면 생성(레이스가 있어도 보통 안전)."""
    try:
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)
    except Exception:
        # 이미 존재 등 예외는 무시(환경에 따라 에러 메시지 다름)
        pass


def cleanup_prefix(minio, bucket: str, prefix: str):
    """job_id/ 아래 기존 crop 오브젝트 삭제(재실행 시 고아 방지)."""
    try:
        for obj in minio.list_objects(bucket, prefix=prefix, recursive=True):
            minio.remove_object(bucket, obj.object_name)
    except Exception:
        # 삭제 실패해도 작업 전체를 죽이지는 않음(원하면 raise로 바꿔도 됨)
        pass


def clamp_bbox(x1, y1, x2, y2, w, h):
    """float bbox를 이미지 범위에 맞게 int로 정리."""
    # 안전하게 floor/ceil
    x1i = int(max(0, min(w - 1, math.floor(x1))))
    y1i = int(max(0, min(h - 1, math.floor(y1))))
    x2i = int(max(0, min(w,     math.ceil(x2))))
    y2i = int(max(0, min(h,     math.ceil(y2))))

    # 최소 1px 보장
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)

    return x1i, y1i, x2i, y2i


def save_crop_and_upload(
    *,
    minio,
    src_img: Image.Image,
    crop_bucket: str,
    object_key: str,
    bbox_xyxy,                 # (x1, y1, x2, y2)
    tmp_dir: str = "/tmp",
    jpeg_quality: int = 90,
) -> tuple[str, tuple[int, int, int, int]]:
    """
    bbox로 crop 이미지 생성 → 임시 저장 → MinIO 업로드 → (object_key, int_bbox) 반환
    """
    w, h = src_img.size
    x1, y1, x2, y2 = bbox_xyxy
    x1i, y1i, x2i, y2i = clamp_bbox(x1, y1, x2, y2, w, h)

    crop = src_img.crop((x1i, y1i, x2i, y2i))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    tmp_path = os.path.join(tmp_dir, f"crop_{uuid.uuid4().hex}.jpg")
    crop.save(tmp_path, format="JPEG", quality=jpeg_quality)

    try:
        minio.fput_object(
            crop_bucket,
            object_key,
            tmp_path,
            content_type="image/jpeg",
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return object_key, (x1i, y1i, x2i, y2i)

