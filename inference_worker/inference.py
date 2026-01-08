import os
from datetime import datetime, timezone
from ultralytics import YOLO

from .config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(settings.MODEL_PATH)
    return _model


def _select_detection(boxes, target, alpha: float):
    if not boxes:
        return None
    tx, ty = target
    contained = []
    for b in boxes:
        conf = b.get("confidence")
        if conf is None:
            continue
        xyxy = b.get("xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        score = conf - alpha * dist
        b = dict(b)
        b["score_weighted"] = score
        if x1 <= tx <= x2 and y1 <= ty <= y2:
            contained.append(b)
    if contained:
        return max(contained, key=lambda b: b["score_weighted"])

    # 포함된 박스가 없으면 모든 박스에 대해 거리-가중 신뢰도로 fallback 선택
    fallback = []
    for b in boxes:
        conf = b.get("confidence")
        if conf is None:
            continue
        xyxy = b.get("xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        score = conf - alpha * dist
        b = dict(b)
        b["score_weighted"] = score
        fallback.append(b)
    return max(fallback, key=lambda b: b["score_weighted"]) if fallback else None


def run_inference_on_file(image_path: str, job_id: str, photo_id: str, rdid: str, img_x: float, img_y: float, alpha: float):
    """
    이미지 파일 경로를 입력으로 받아 YOLO 세그먼트 추론을 수행하고
    결과 딕셔너리와 주석 이미지를 저장한 로컬 경로(없으면 None)를 반환한다.
    """
    predictor = _get_model()
    preds = predictor(image_path, verbose=False, conf=0.7)
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

    selected = _select_detection(boxes, (img_x, img_y), alpha)
    result["selected"] = selected

    annotated_path = None
    if not no_detections:
        os.makedirs(settings.TMP_DIR, exist_ok=True)
        annotated_path = os.path.join(settings.TMP_DIR, f"{job_id}_pred.jpg")
        pred.save(filename=annotated_path)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result, annotated_path
