from __future__ import annotations

import re
from typing import Any

FULL_FAMILY_TARGETS = frozenset(
    {
        "109",
        "216",
        "217",
        "218",
        "219",
        "220",
        "221",
        "401",
        "402",
        "403",
        "404",
        "405",
        "406",
        "408",
        "409",
        "410",
        "411",
        "413",
        "414",
        "415",
        "420",
        "421",
        "422",
        "423",
        "424",
        "425",
        "426",
        "427",
        "428",
        "431",
        "433",
        "435",
        "436",
        "440",
        "441",
        "442",
        "443",
        "444",
        "445",
        "448",
        "501",
        "504",
        "505",
        "507",
        "513",
        "514",
        "516",
        "517",
        "518",
        "901",
        "907",
    }
)

MIXED_FAMILY_EXACT_TARGETS: dict[str, frozenset[str]] = {
    "407": frozenset({"407_1", "407_4"}),
    "412": frozenset({"412_1"}),
    "430": frozenset({"430_3"}),
    "434": frozenset({"434_1"}),
    "446": frozenset({"446_1"}),
    "499": frozenset({"49998", "49998_1", "49999"}),
    "515": frozenset({"515_2"}),
    "999": frozenset({"99900", "99900_8", "99999"}),
}

TARGET_CATEGORY_BY_FAMILY: dict[str, str] = {
    "109": "주의표지",
    "216": "규제표지",
    "217": "규제표지",
    "218": "규제표지",
    "219": "규제표지",
    "220": "규제표지",
    "221": "규제표지",
    "401": "경계표지",
    "402": "이정표지",
    "403": "방향표지",
    "404": "노선표지",
    "405": "휴게소표지",
    "406": "관광지표지",
    "407": "양보차로표지",
    "408": "유도표지",
    "409": "예고표지",
    "410": "방향표지",
    "411": "안내기타표지",
    "412": "안내기타표지",
    "413": "이정표지",
    "414": "노선표지",
    "415": "교통기타표지",
    "420": "경계표지",
    "421": "이정표지",
    "422": "예고표지",
    "423": "예고표지",
    "424": "예고표지",
    "425": "방향표지",
    "426": "노선표지",
    "427": "안내기타표지",
    "428": "휴게소표지",
    "430": "예고표지",
    "431": "노선표지",
    "433": "예고표지",
    "434": "오르막차로표지",
    "435": "유도표지",
    "436": "안내기타표지",
    "440": "도로명 방향표지",
    "441": "이정표지",
    "442": "경계표지",
    "443": "노선표지",
    "444": "안내기타표지",
    "445": "관광지표지",
    "446": "안내기타표지",
    "448": "안내기타표지",
    "499": "안내기타표지",
    "501": "보조표지",
    "504": "보조표지",
    "505": "보조표지",
    "507": "보조표지",
    "513": "보조표지",
    "514": "보조표지",
    "515": "보조표지",
    "516": "보조표지",
    "517": "보조표지",
    "518": "보조표지",
    "901": "교통기타표지",
    "907": "교통기타표지",
    "999": "보조표지",
}

CATEGORY_TO_SHAPE: dict[str, str] = {
    "주의표지": "triangle",
    "규제표지": "circle",
    "지시표지": "circle",
    "경계표지": "rectangle",
    "이정표지": "rectangle",
    "방향표지": "rectangle",
    "노선표지": "rectangle",
    "휴게소표지": "rectangle",
    "관광지표지": "rectangle",
    "양보차로표지": "rectangle",
    "오르막차로표지": "rectangle",
    "유도표지": "rectangle",
    "예고표지": "rectangle",
    "안내기타표지": "rectangle",
    "교통기타표지": "rectangle",
    "도로명 방향표지": "rectangle",
    "보조표지": "rectangle",
    "기타표지": "other",
}

_FAMILY_RE = re.compile(r"(\d{3})")


def normalize_class_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def class_family(value: Any) -> str | None:
    normalized = normalize_class_name(value)
    if not normalized:
        return None
    match = _FAMILY_RE.search(normalized)
    if not match:
        return None
    return match.group(1)


def resolve_ocr_policy(class_name: Any) -> dict[str, Any]:
    normalized = normalize_class_name(class_name)
    family = class_family(normalized)
    category = TARGET_CATEGORY_BY_FAMILY.get(family)
    shape = CATEGORY_TO_SHAPE.get(category, "other")

    enabled = False
    ambiguous = False
    source = "non_target"

    if family in FULL_FAMILY_TARGETS:
        enabled = True
        source = "full_family"
    else:
        exact_targets = MIXED_FAMILY_EXACT_TARGETS.get(family)
        if exact_targets:
            if normalized in exact_targets:
                enabled = True
                source = "mixed_exact"
            else:
                ambiguous = normalized == family
                source = "mixed_ambiguous_family" if ambiguous else "mixed_non_target"

    return {
        "enabled": enabled,
        "ambiguous": ambiguous,
        "source": source,
        "normalized_class_name": normalized,
        "class_family": family,
        "category": category,
        "shape": shape,
    }


def annotate_crop_item(item: dict[str, Any]) -> dict[str, Any]:
    merged = dict(item)
    policy = resolve_ocr_policy(merged.get("class_name"))
    merged["ocr_target"] = policy["enabled"]
    merged["ocr_target_ambiguous"] = policy["ambiguous"]
    merged["ocr_policy_source"] = policy["source"]
    merged["ocr_class_name_normalized"] = policy["normalized_class_name"]
    merged["ocr_class_family"] = policy["class_family"]
    merged["ocr_category"] = policy["category"]
    merged["ocr_shape"] = policy["shape"]
    return merged


def _localize_mask_polygon(mask_polygon: Any, bbox_xyxy: Any) -> list[list[float]] | None:
    if not isinstance(mask_polygon, (list, tuple)) or len(mask_polygon) < 3:
        return None
    if not (isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4):
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    except Exception:
        return None

    crop_w = max(1.0, x2 - x1)
    crop_h = max(1.0, y2 - y1)
    localized: list[list[float]] = []

    for point in mask_polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            px = float(point[0]) - x1
            py = float(point[1]) - y1
        except Exception:
            continue

        px = max(0.0, min(crop_w - 1.0, px))
        py = max(0.0, min(crop_h - 1.0, py))
        localized.append([px, py])

    return localized if len(localized) >= 3 else None


def build_ocr_queue_items(
    crop_items: list[dict[str, Any]] | None,
    masks: list[Any] | None = None,
) -> list[dict[str, Any]]:
    items = crop_items or []
    mask_list = masks or []
    enriched: list[dict[str, Any]] = []

    for item in items:
        merged = annotate_crop_item(item)
        det_index = merged.get("det_index")
        try:
            det_index_int = int(det_index)
        except Exception:
            det_index_int = None
        if det_index_int is not None and 0 <= det_index_int < len(mask_list):
            localized = _localize_mask_polygon(mask_list[det_index_int], merged.get("bbox_xyxy"))
            if localized:
                merged["mask_polygon"] = localized
        enriched.append(merged)

    return enriched


def filter_ocr_target_items(
    crop_items: list[dict[str, Any]] | None,
    masks: list[Any] | None = None,
) -> list[dict[str, Any]]:
    return [item for item in build_ocr_queue_items(crop_items, masks) if item.get("ocr_target")]
