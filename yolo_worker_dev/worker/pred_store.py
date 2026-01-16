# yolo_worker/pred_store.py
import os
import uuid
import numpy as np
from PIL import Image

def save_pred_and_upload(
    *,
    minio,
    result,                 # ultralytics Results (results[0])
    pred_bucket: str,
    object_key: str,
    tmp_dir: str = "/tmp",
    jpeg_quality: int = 90,
) -> str:
    """
    Ultralytics result를 시각화(박스/마스크 오버레이)해서 JPG로 저장 후 MinIO 업로드.
    반환: object_key
    """
    # result.plot() -> numpy ndarray(BGR)인 경우가 일반적
    im = result.plot()

    if isinstance(im, np.ndarray):
        # 보통 BGR 3채널 -> RGB로 변환
        if im.ndim == 3 and im.shape[2] == 3:
            im = im[..., ::-1]
        pil = Image.fromarray(im)
    else:
        # 혹시 PIL 이미지로 오는 경우 대비
        pil = im

    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    tmp_path = os.path.join(tmp_dir, f"pred_{uuid.uuid4().hex}.jpg")
    pil.save(tmp_path, format="JPEG", quality=jpeg_quality)

    try:
        minio.fput_object(
            pred_bucket,
            object_key,
            tmp_path,
            content_type="image/jpeg",
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return object_key

