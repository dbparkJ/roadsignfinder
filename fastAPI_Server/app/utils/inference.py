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


def normalize_bbox(bbox: list | tuple | None) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        return None


def bbox_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def select_best_bbox_match(
    items: list,
    target_bbox: tuple[float, float, float, float] | None,
    bbox_getter,
):
    if not isinstance(items, list) or target_bbox is None:
        return None

    target_cx = (target_bbox[0] + target_bbox[2]) / 2.0
    target_cy = (target_bbox[1] + target_bbox[3]) / 2.0
    best = None
    for item in items:
        candidate = bbox_getter(item)
        if candidate is None:
            continue
        iou = bbox_iou(target_bbox, candidate)
        dist = bbox_center_distance_sq(target_cx, target_cy, candidate)
        score = (iou, -dist)
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best is not None else None


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

    def _compact_box(box: dict | None) -> dict | None:
        if not isinstance(box, dict):
            return None
        return {
            "xyxy": box.get("xyxy"),
            "class_id": box.get("class_id"),
            "class_name": box.get("class_name"),
            "confidence": box.get("confidence"),
        }

    def _compact_crop_item(item: dict | None) -> dict | None:
        if not isinstance(item, dict):
            return None
        return {
            "bbox_xyxy": item.get("bbox_xyxy"),
            "class_id": item.get("class_id"),
            "class_name": item.get("class_name"),
            "confidence": item.get("confidence"),
        }

    def _extract_ocr_texts_from_item(item: dict | None) -> list[str]:
        if not isinstance(item, dict):
            return []
        texts = item.get("detected_texts")
        if isinstance(texts, list):
            return [str(text).strip() for text in texts if str(text).strip()]

        extracted: list[str] = []
        for page in item.get("pages") or []:
            if not isinstance(page, dict):
                continue
            page_texts = page.get("detected_texts")
            if isinstance(page_texts, list):
                for text in page_texts:
                    text = str(text).strip()
                    if text and text not in extracted:
                        extracted.append(text)
                continue
            for value in page.get("json_values") or []:
                if not isinstance(value, dict):
                    continue
                for block in value.get("parsing_res_list") or []:
                    if isinstance(block, dict):
                        text = str(block.get("block_content", "")).strip()
                        if text and text not in extracted:
                            extracted.append(text)
        return extracted

    def _compact_pole_type_payload(pole_type: dict | None) -> dict | None:
        if not isinstance(pole_type, dict):
            return None
        result = pole_type.get("result_json")
        labels = result.get("labels") if isinstance(result, dict) else None
        if isinstance(labels, list) and target_bbox is not None:
            best_label = select_best_bbox_match(
                labels,
                target_bbox,
                lambda label: normalize_bbox(label.get("sign_box")) if isinstance(label, dict) else None,
            )
            labels = [best_label] if best_label is not None else []
        compacted = {
            "status": pole_type.get("status"),
            "labels": [],
        }
        if isinstance(labels, list):
            compacted["labels"] = [
                {
                    "label_code": label.get("label_code"),
                    "label_name": label.get("label_name"),
                }
                for label in labels
                if isinstance(label, dict)
            ]
        if pole_type.get("error_message"):
            compacted["error_message"] = pole_type.get("error_message")
        return compacted

    def _compact_ocr_payload(ocr: dict | None) -> dict | None:
        if not isinstance(ocr, dict):
            return None
        result = ocr.get("result_json")
        items = result.get("items") if isinstance(result, dict) else None
        if isinstance(items, list) and target_bbox is not None:
            best_item = select_best_bbox_match(
                items,
                target_bbox,
                lambda item: normalize_bbox(item.get("bbox_xyxy")) if isinstance(item, dict) else None,
            )
            items = [best_item] if best_item is not None else []
        compacted = {
            "status": ocr.get("status"),
            "items": [],
        }
        if isinstance(result, dict) and result.get("reason"):
            compacted["reason"] = result.get("reason")
        if isinstance(items, list):
            compacted["items"] = [
                {
                    "bbox_xyxy": item.get("bbox_xyxy"),
                    "class_id": item.get("class_id"),
                    "class_name": item.get("class_name"),
                    "confidence": item.get("confidence"),
                    "detected_texts": _extract_ocr_texts_from_item(item),
                }
                for item in items
                if isinstance(item, dict)
            ]
        if ocr.get("error_message"):
            compacted["error_message"] = ocr.get("error_message")
        return compacted

    wrapper = result_json if "yolo" in result_json else {"yolo": result_json}
    yolo_payload = wrapper.get("yolo")
    if not isinstance(yolo_payload, dict):
        return result_json

    target_bbox = None
    boxes = yolo_payload.get("boxes") or []
    masks = yolo_payload.get("masks") or []
    if len(boxes) == 1 and isinstance(boxes[0], dict):
        target_bbox = normalize_bbox(boxes[0].get("xyxy"))
    if target_bbox is None and len(masks) == 1 and isinstance(masks[0], list):
        target_bbox = bbox_from_poly(masks[0])

    compacted = {
        "boxes": [],
        "masks": [],
        "crop_images": [],
    }
    compacted["boxes"] = [
        item
        for item in (_compact_box(box) for box in (yolo_payload.get("boxes") or []))
        if item is not None
    ]
    compacted["masks"] = [
        mask for mask in (yolo_payload.get("masks") or []) if isinstance(mask, list)
    ]
    crop_images = yolo_payload.get("crop_images") or []
    if isinstance(crop_images, list) and target_bbox is not None:
        best_crop = select_best_bbox_match(
            crop_images,
            target_bbox,
            lambda item: normalize_bbox(item.get("bbox_xyxy")) if isinstance(item, dict) else None,
        )
        crop_images = [best_crop] if best_crop is not None else []
    compacted["crop_images"] = [
        item
        for item in (_compact_crop_item(crop) for crop in crop_images)
        if item is not None
    ]

    pole_type = _compact_pole_type_payload(wrapper.get("pole_type"))
    if pole_type is not None:
        compacted["pole_type"] = pole_type

    ocr = _compact_ocr_payload(wrapper.get("ocr"))
    if ocr is not None:
        compacted["ocr"] = ocr

    return compacted


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
    def _select_matching_crop_images(crop_images: list, box_xyxy: list | tuple | None) -> list:
        if not isinstance(crop_images, list):
            return []
        target = normalize_bbox(box_xyxy)
        if target is None:
            return []
        best = select_best_bbox_match(
            crop_images,
            target,
            lambda item: normalize_bbox(item.get("bbox_xyxy")) if isinstance(item, dict) else None,
        )
        return [best] if best is not None else []

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
    payload["crop_images"] = _select_matching_crop_images(payload.get("crop_images") or [], box)
    if det.get("confidence") is not None:
        payload["scores"] = [det.get("confidence")]
    return payload
