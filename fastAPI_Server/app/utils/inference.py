from __future__ import annotations


def point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float, tol: float) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return (x - x1) * (x - x1) + (y - y1) * (y - y1) <= tol * tol
    t = ((x - x1) * dx + (y - y1) * dy) / denom
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= tol * tol


def point_in_poly(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if point_on_segment(x, y, xi, yi, xj, yj, tol=1.0):
            return True
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_in_masks(x: float, y: float, masks: list[list[list[float]]]) -> bool:
    for poly in masks:
        if point_in_poly(x, y, poly):
            return True
    return False


def bbox_from_poly(poly: list[list[float]]) -> tuple[float, float, float, float] | None:
    if not poly:
        return None
    xs = [p[0] for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [p[1] for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def bbox_contains_point(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_center_distance_sq(x: float, y: float, bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = x - cx
    dy = y - cy
    return dx * dx + dy * dy


def select_nearest_yolo_payload(
    yolo_payload: dict,
    img_x: float,
    img_y: float,
) -> dict | None:
    if not isinstance(yolo_payload, dict):
        return None

    candidates: list[dict] = []
    masks = yolo_payload.get("masks") or []
    for idx, poly in enumerate(masks):
        bbox = bbox_from_poly(poly) if isinstance(poly, list) else None
        if bbox:
            candidates.append({"kind": "mask", "index": idx, "bbox": bbox})

    boxes = yolo_payload.get("boxes") or []
    for idx, box in enumerate(boxes):
        xyxy = box.get("xyxy") if isinstance(box, dict) else None
        if isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4:
            bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            candidates.append({"kind": "box", "index": idx, "bbox": bbox})

    if not candidates:
        return None

    best = None
    for cand in candidates:
        if bbox_contains_point(img_x, img_y, cand["bbox"]):
            dist = bbox_center_distance_sq(img_x, img_y, cand["bbox"])
            if best is None or dist < best["dist"]:
                best = {"dist": dist, **cand}

    if not best:
        return None

    selected = dict(yolo_payload)
    if best["kind"] == "mask":
        if isinstance(masks, list) and 0 <= best["index"] < len(masks):
            selected["masks"] = [masks[best["index"]]]
        if isinstance(boxes, list) and 0 <= best["index"] < len(boxes):
            selected["boxes"] = [boxes[best["index"]]]
    elif best["kind"] == "box":
        if isinstance(boxes, list) and 0 <= best["index"] < len(boxes):
            selected["boxes"] = [boxes[best["index"]]]
        if isinstance(masks, list) and 0 <= best["index"] < len(masks):
            selected["masks"] = [masks[best["index"]]]

    return selected


def select_nearest_result_json(
    result_json: dict | None,
    img_x: float,
    img_y: float,
) -> dict | str | None:
    if not isinstance(result_json, dict):
        return result_json

    yolo_payload = result_json.get("yolo")
    has_wrapper = isinstance(yolo_payload, dict)
    if not has_wrapper:
        yolo_payload = result_json

    if not isinstance(yolo_payload, dict):
        return result_json

    selected = select_nearest_yolo_payload(yolo_payload, img_x, img_y)
    if not selected:
        return "None"

    if has_wrapper:
        merged = dict(result_json)
        merged["yolo"] = selected
        return merged
    return selected


def compact_inference_result_json(result_json: dict | None) -> dict | None:
    if not isinstance(result_json, dict):
        return result_json
    if "yolo" in result_json:
        yolo_payload = result_json.get("yolo")
        if isinstance(yolo_payload, dict):
            merged = dict(yolo_payload)
            pole_type = result_json.get("pole_type")
            if pole_type is not None:
                merged["pole_type"] = pole_type
            return merged
    return result_json


def mask_bbox(mask: list | None) -> tuple[float, float, float, float] | None:
    if not isinstance(mask, list) or not mask:
        return None
    return bbox_from_poly(mask)


def select_nearest_detection(detections: list[dict], img_x: float, img_y: float) -> dict | None:
    if not detections:
        return None
    best = None
    for det in detections:
        bbox = det.get("box_xyxy")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            bbox = mask_bbox(det.get("mask"))
        if not bbox:
            continue
        bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
        if bbox_contains_point(img_x, img_y, bbox):
            dist = bbox_center_distance_sq(img_x, img_y, bbox)
            if best is None or dist < best["dist"]:
                best = {"dist": dist, "det": det, "bbox": bbox}
    if not best:
        return None
    return best["det"]


def build_selected_yolo_payload(base_yolo: dict | None, det: dict) -> dict:
    payload = dict(base_yolo or {})
    box = det.get("box_xyxy")
    mask = det.get("mask")
    payload["boxes"] = [
        {
            "xyxy": box,
            "class_id": det.get("class_id"),
            "class_name": det.get("class_name"),
            "confidence": det.get("confidence"),
        }
    ] if box is not None else []
    payload["masks"] = [mask] if mask is not None else []
    if det.get("confidence") is not None:
        payload["scores"] = [det.get("confidence")]
    return payload
