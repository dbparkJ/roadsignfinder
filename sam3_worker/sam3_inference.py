import os
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
from ultralytics import YOLO

from .config import settings


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_sam3_model = None
_sam3_processor = None
_yolo_model = None

LABEL_MAP = {
    1401: "단주식",
    1402: "복주식",
    1403: "측주식",
    1404: "편지식",
    1405: "복합식",
    1406: "문형식",
    1407: "내민식",
    1408: "부착식",
    1409: "현수식",
    1499: "기타",
}


def _get_sam3():
    global _sam3_model, _sam3_processor
    if _sam3_model is None or _sam3_processor is None:
        _sam3_model = Sam3Model.from_pretrained(settings.SAM3_MODEL_NAME).to(DEVICE)
        _sam3_processor = Sam3Processor.from_pretrained(settings.SAM3_MODEL_NAME)
        if settings.SAM3_USE_FP16 and DEVICE == "cuda":
            _sam3_model = _sam3_model.half()
    return _sam3_model, _sam3_processor


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(settings.YOLO_MODEL_PATH)
    return _yolo_model


def _clamp_box(x1, y1, x2, y2, w, h):
    x1p = max(0, int(x1))
    y1p = max(0, int(y1))
    x2p = min(w - 1, int(x2))
    y2p = min(h - 1, int(y2))
    if x2p <= x1p:
        x2p = min(w - 1, x1p + 1)
    if y2p <= y1p:
        y2p = min(h - 1, y1p + 1)
    return x1p, y1p, x2p, y2p


def _mask_to_rgba(mask_bin, color_rgb, alpha=120):
    colored = np.zeros((*mask_bin.shape, 4), dtype=np.uint8)
    colored[mask_bin > 0] = (*color_rgb, alpha)
    return Image.fromarray(colored, mode="RGBA")


def _union_masks(masks):
    if masks is None or len(masks) == 0:
        return None
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    return out


def _count_vertical_poles(mask_union):
    if mask_union is None:
        return 0
    col = mask_union.sum(axis=0).astype(np.float32)
    if col.max() <= 0:
        return 0
    col = col / (col.max() + 1e-6)
    thr = 0.35
    above = col > thr
    peaks = 0
    in_seg = False
    for v in above:
        if v and not in_seg:
            peaks += 1
            in_seg = True
        elif not v and in_seg:
            in_seg = False
    return peaks


def _select_horizontal_masks(masks, aspect_min=3.0):
    sel = []
    for m in masks:
        mb = (m > 0).astype(np.uint8)
        ys, xs = np.where(mb > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        w = max(1, x2 - x1 + 1)
        h = max(1, y2 - y1 + 1)
        if (w / float(h)) >= aspect_min:
            sel.append(mb)
    return sel


def _classify_pole_shape(sign_box, vert_union, horz_union):
    sx1, sy1, sx2, sy2 = sign_box
    sign_w = max(1, sx2 - sx1 + 1)
    sign_cx = 0.5 * (sx1 + sx2)

    vcnt = _count_vertical_poles(vert_union)
    has_horz = horz_union is not None and horz_union.sum() > 200

    if vert_union is not None and vert_union.sum() > 0:
        ys, xs = np.where(vert_union > 0)
        if len(ys) > 0:
            vert_top = int(ys.min())
            if vert_top < (sy1 - 5):
                return 1408
            pole_cx = float(xs.mean())
        else:
            pole_cx = sign_cx
    else:
        pole_cx = sign_cx

    offset = abs(pole_cx - sign_cx) / float(sign_w)

    if (vert_union is None or vert_union.sum() < 200) and not has_horz:
        return 1408
    if vcnt >= 2 and has_horz:
        return 1408
    if vcnt >= 2 and not has_horz:
        return 1402
    if vcnt == 1 and has_horz:
        return 1408
    if vcnt == 1 and not has_horz:
        return 1403 if offset > 0.35 else 1401
    if vcnt == 0 and has_horz:
        return 1408
    return 1499


def _run_sam3_text(model, processor, image_pil, text):
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
    if settings.SAM3_USE_FP16 and DEVICE == "cuda":
        inputs = {k: (v.half() if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results["masks"]
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    return masks


def _filter_masks_by_area(masks, min_area):
    kept = []
    for m in masks:
        if np.sum(m > 0) >= min_area:
            kept.append(m)
    return kept


def run_sam3_on_file(
    image_path: str,
    job_id: str,
    photo_id: str,
    rdid: str,
    overlay_base_path: str | None = None,
):
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[pole_type] input_image={image_path}")
    model, processor = _get_sam3()
    yolo = _get_yolo()

    image = Image.open(image_path).convert("RGB")
    image_rgb = image.copy()
    overlay = image_rgb.copy()
    if overlay_base_path:
        try:
            base_img = Image.open(overlay_base_path).convert("RGB")
            if base_img.size != image_rgb.size:
                base_img = base_img.resize(image_rgb.size)
            overlay = base_img.copy()
        except Exception:
            overlay = image_rgb.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image_rgb.size

    yolo_results = yolo(image, verbose=False, conf=settings.YOLO_CONF)[0]
    boxes_xyxy = []
    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        result = {
            "job_id": job_id,
            "photo_id": photo_id,
            "rdid": rdid,
            "model": settings.SAM3_MODEL_NAME,
            "labels": [],
            "no_detections": True,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        print("[pole_type] result_image=none")
        return result, None

    sign_masks = None
    if getattr(yolo_results, "masks", None) is not None and yolo_results.masks is not None:
        sign_masks = yolo_results.masks.data.cpu().numpy()

    labels = []
    use_overlay_base = overlay_base_path is not None
    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, width, height)
        sign_w = max(1, x2 - x1 + 1)
        sign_h = max(1, y2 - y1 + 1)

        v_strip_w = int(sign_w * settings.V_STRIP_SCALE)
        h_strip_h = int(sign_h * settings.H_STRIP_SCALE)

        vx1 = int(0.5 * (x1 + x2) - 0.5 * v_strip_w)
        vx2 = vx1 + v_strip_w
        hx1 = 0
        hx2 = width - 1

        vy1 = 0
        vy2 = height - 1
        hy1 = int(0.5 * (y1 + y2) - 0.5 * h_strip_h)
        hy2 = hy1 + h_strip_h

        vx1, vy1, vx2, vy2 = _clamp_box(vx1, vy1, vx2, vy2, width, height)
        hx1, hy1, hx2, hy2 = _clamp_box(hx1, hy1, hx2, hy2, width, height)

        v_strip = image_rgb.crop((vx1, vy1, vx2 + 1, vy2 + 1))
        h_strip = image_rgb.crop((hx1, hy1, hx2 + 1, hy2 + 1))

        v_masks = _filter_masks_by_area(
            _run_sam3_text(model, processor, v_strip, text="pole"),
            settings.MIN_MASK_AREA,
        )
        h_masks = _filter_masks_by_area(
            _run_sam3_text(model, processor, h_strip, text="pole"),
            settings.MIN_MASK_AREA,
        )

        v_union = _union_masks([(m > 0).astype(np.uint8) for m in v_masks])
        h_sel = _select_horizontal_masks(h_masks, aspect_min=3.0)
        h_union = _union_masks(h_sel)

        if v_union is not None:
            v_full = np.zeros((height, width), dtype=np.uint8)
            v_full[vy1:vy2 + 1, vx1:vx2 + 1] = v_union
        else:
            v_full = None

        if h_union is not None:
            h_full = np.zeros((height, width), dtype=np.uint8)
            h_full[hy1:hy2 + 1, hx1:hx2 + 1] = h_union
        else:
            h_full = None

        label_code = _classify_pole_shape((x1, y1, x2, y2), v_full, h_full)
        label_name = LABEL_MAP.get(label_code, LABEL_MAP[1499])
        labels.append(
            {
                "label_code": label_code,
                "label_name": label_name,
                "sign_box": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

        overlay_rgba = overlay.convert("RGBA")
        if not use_overlay_base:
            if sign_masks is not None and idx < len(sign_masks):
                sign_mask = (sign_masks[idx] > 0.5).astype(np.uint8)
                sign_mask_img = _mask_to_rgba(sign_mask, (0, 255, 0), alpha=80)
                overlay_rgba.paste(sign_mask_img, (0, 0), sign_mask_img)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.text((x1 + 4, y1 + 4), f"{label_code} {label_name}", fill=(0, 255, 0))

        if v_full is not None:
            v_mask = _mask_to_rgba(v_full, (255, 0, 0), alpha=80)
            overlay_rgba.paste(v_mask, (0, 0), v_mask)
        if h_full is not None:
            h_mask = _mask_to_rgba(h_full, (0, 120, 255), alpha=70)
            overlay_rgba.paste(h_mask, (0, 0), h_mask)
        overlay = overlay_rgba.convert("RGB")
        draw = ImageDraw.Draw(overlay)

    result = {
        "job_id": job_id,
        "photo_id": photo_id,
        "rdid": rdid,
        "model": settings.SAM3_MODEL_NAME,
        "labels": labels,
        "no_detections": len(labels) == 0,
        "started_at": started_at,
    }
    result["finished_at"] = datetime.now(timezone.utc).isoformat()

    annotated_path = None
    if labels:
        os.makedirs(settings.TMP_DIR, exist_ok=True)
        annotated_path = os.path.join(settings.TMP_DIR, f"{job_id}_sam3.jpg")
        overlay.save(annotated_path)

    print(f"[pole_type] result_image={annotated_path if annotated_path else 'none'}")
    return result, annotated_path
