import os
import time
from datetime import datetime, timezone

from PIL import Image
from imgutils.restore import restore_with_nafnet
from ultralytics import YOLO

from .config import settings

_model = None


def _debug(msg: str) -> None:
    if settings.INFERENCE_DEBUG_LOG:
        print(msg)


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"model file not found: {settings.MODEL_PATH}")
        _model = YOLO(settings.MODEL_PATH)
    return _model


def _save_crop_source_image(image: Image.Image, crop_source_path: str | None) -> str | None:
    if not crop_source_path:
        return None
    out_path = os.path.abspath(os.path.expanduser(crop_source_path))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_image = image.convert("RGB") if image.mode != "RGB" else image
    save_image.save(out_path, format="JPEG", quality=95)
    return out_path


def run_inference_on_file(
    image_path: str,
    job_id: str,
    photo_id: str,
    rdid: str,
    img_x: float,
    img_y: float,
    crop_source_path: str | None = None,
):
    """
    이미지 파일 경로를 입력으로 받아 YOLO 세그먼트 추론을 수행하고
    결과 딕셔너리, 주석 이미지 경로, crop 생성에 사용할 입력 이미지 경로를 반환한다.
    """
    t0 = time.perf_counter()
    predictor = _get_model()
    t1 = time.perf_counter()
    _debug(f"[inference] stage=load_model_done sec={t1 - t0:.3f}")
    image = Image.open(image_path).convert("RGB")
    try:
        t_restore0 = time.perf_counter()
        image = restore_with_nafnet(image, model="REDS")
        t_restore1 = time.perf_counter()
        _debug(f"[inference] stage=restore_done sec={t_restore1 - t_restore0:.3f}")
    except Exception:
        _debug("[inference] stage=restore_skipped")
        pass
    t2 = time.perf_counter()
    preds = predictor(image, verbose=False, conf=0.6)
    t3 = time.perf_counter()
    _debug(f"[inference] stage=yolo_done sec={t3 - t2:.3f}")
    pred = preds[0]

    boxes = []
    masks = []
    scores = []

    names = pred.names if hasattr(pred, "names") else {}
    if pred.boxes is not None and len(pred.boxes) > 0:
        for b in pred.boxes:
            cls_id = int(b.cls.item()) if b.cls is not None else -1
            boxes.append(
                {
                    "xyxy": [float(x) for x in b.xyxy[0].tolist()],
                    "confidence": float(b.conf.item()) if b.conf is not None else None,
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                }
            )
            scores.append(float(b.conf.item()) if b.conf is not None else None)

    if pred.masks is not None and pred.masks.xy is not None:
        for poly in pred.masks.xy:
            masks.append([[float(x), float(y)] for x, y in poly.tolist()])

    result = {
        "job_id": job_id,
        "photo_id": photo_id,
        "rdid": rdid,
        "model": settings.MODEL_NAME,
        "boxes": boxes,
        "masks": masks,
        "scores": scores,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    no_detections = not boxes and not masks and not scores
    result["no_detections"] = no_detections

    annotated_path = None
    used_crop_source_path = image_path
    if not no_detections:
        os.makedirs(settings.TMP_DIR, exist_ok=True)
        annotated_path = os.path.join(settings.TMP_DIR, f"{job_id}_pred.jpg")
        pred.save(filename=annotated_path)
        try:
            saved_crop_source = _save_crop_source_image(image, crop_source_path)
            if saved_crop_source:
                used_crop_source_path = saved_crop_source
                _debug(f"[inference] crop_source_saved path={used_crop_source_path}")
        except Exception as e:
            _debug(f"[inference] crop_source_save_failed error={e}")

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result, annotated_path, used_crop_source_path
