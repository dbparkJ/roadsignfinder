from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .config import settings


def _bicubic():
    return Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def _lanczos():
    return Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _quad_transform():
    return Image.Transform.QUAD if hasattr(Image, "Transform") else Image.QUAD


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _quad_from_polygon(points: list[list[float]]) -> np.ndarray | None:
    if len(points) < 4:
        return None

    pts = np.asarray(points, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]

    quad = np.asarray([tl, tr, br, bl], dtype=np.float32)
    uniq = {tuple(np.round(p, 2)) for p in quad}
    if len(uniq) < 4:
        return None
    return quad


def _rectify_rectangle(img: Image.Image, polygon: list[list[float]]) -> Image.Image | None:
    quad = _quad_from_polygon(polygon)
    if quad is None:
        return None

    tl, tr, br, bl = quad
    out_w = max(32, int(round(max(_distance(tl, tr), _distance(bl, br)))))
    out_h = max(32, int(round(max(_distance(tl, bl), _distance(tr, br)))))

    data = (
        float(tl[0]),
        float(tl[1]),
        float(bl[0]),
        float(bl[1]),
        float(br[0]),
        float(br[1]),
        float(tr[0]),
        float(tr[1]),
    )
    try:
        return img.transform((out_w, out_h), _quad_transform(), data, resample=_bicubic())
    except Exception:
        return None


def _polygon_bbox(
    polygon: list[list[float]],
    width: int,
    height: int,
    *,
    pad_ratio: float = 0.08,
    square: bool = False,
) -> tuple[int, int, int, int] | None:
    if len(polygon) < 3:
        return None

    pts = np.asarray(polygon, dtype=np.float32)
    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))

    w = max(1.0, max_x - min_x)
    h = max(1.0, max_y - min_y)
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    if square:
        side = max(max_x - min_x, max_y - min_y)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        half = side / 2.0
        min_x = cx - half
        max_x = cx + half
        min_y = cy - half
        max_y = cy + half

    left = max(0, int(math.floor(min_x)))
    upper = max(0, int(math.floor(min_y)))
    right = min(width, int(math.ceil(max_x)))
    lower = min(height, int(math.ceil(max_y)))

    if right <= left or lower <= upper:
        return None
    return (left, upper, right, lower)


def _pad_and_upscale(img: Image.Image, *, min_edge: int = 160, pad_ratio: float = 0.06) -> Image.Image:
    border = max(4, int(round(max(img.size) * pad_ratio)))
    padded = _expand_with_background(img, border)
    cur_min = min(padded.size)
    if cur_min >= min_edge:
        return padded

    scale = float(min_edge) / float(max(1, cur_min))
    out_w = max(1, int(round(padded.width * scale)))
    out_h = max(1, int(round(padded.height * scale)))
    return padded.resize((out_w, out_h), resample=_lanczos())


def _expand_with_background(img: Image.Image, border_px: int) -> Image.Image:
    if border_px <= 0:
        return img.copy()

    arr = np.asarray(img)
    if arr.ndim == 2:
        padded = np.pad(
            arr,
            ((border_px, border_px), (border_px, border_px)),
            mode="edge",
        )
    else:
        padded = np.pad(
            arr,
            ((border_px, border_px), (border_px, border_px), (0, 0)),
            mode="edge",
        )
    return Image.fromarray(padded)


def _expand_padding(img: Image.Image, extra_padding_px: int) -> Image.Image:
    if extra_padding_px <= 0:
        return img.copy()
    return _expand_with_background(img, extra_padding_px)


def _enhance_small_crop(
    img: Image.Image,
    *,
    source_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, bool]:
    threshold = max(0, int(settings.OCR_SMALL_CROP_DOUBLE_EDGE_THRESHOLD))
    scale = float(settings.OCR_SMALL_CROP_DOUBLE_SCALE)
    if threshold <= 0 or scale <= 1.0:
        return img, False
    ref_w, ref_h = source_size or img.size
    if ref_w >= threshold and ref_h >= threshold:
        return img, False

    out_w = max(1, int(round(img.width * scale)))
    out_h = max(1, int(round(img.height * scale)))
    enlarged = img.resize((out_w, out_h), resample=_lanczos())
    # Small crops tend to be jagged after enlargement, so apply a mild smoothing pass
    # before the final OCR padding/upscale stage.
    return enlarged.filter(ImageFilter.SMOOTH), True


def _safe_group_name(crop: dict[str, Any], crop_path: Path) -> str:
    det_index = crop.get("det_index")
    try:
        return f"det_{int(det_index):03d}"
    except Exception:
        return crop_path.stem


def _prepare_base_image(crop: dict[str, Any]) -> tuple[Image.Image, str, Path, tuple[int, int]]:
    crop_path = Path(str(crop.get("crop_path", ""))).expanduser().resolve()
    shape = str(crop.get("ocr_shape") or "other")
    polygon = crop.get("mask_polygon") if isinstance(crop.get("mask_polygon"), list) else None

    with Image.open(crop_path) as opened:
        src = opened.convert("RGB")
        source_size = src.size
        prepared: Image.Image | None = None
        mode = "passthrough"

        if polygon and shape == "rectangle":
            prepared = _rectify_rectangle(src, polygon)
            if prepared is not None:
                mode = "rectify_quad"

        if prepared is None and polygon:
            bbox = _polygon_bbox(
                polygon,
                src.width,
                src.height,
                square=shape == "circle",
            )
            if bbox is not None:
                prepared = src.crop(bbox)
                mode = "tight_crop_circle" if shape == "circle" else "tight_crop_polygon"

        if prepared is None:
            prepared = src.copy()

    return prepared, mode, crop_path, source_size


def _build_variant_item(
    crop: dict[str, Any],
    *,
    base_image: Image.Image,
    preprocess_mode: str,
    crop_path: Path,
    output_dir: Path,
    variant_label: str,
    padding_px: int,
    small_crop_enhanced: bool,
) -> dict[str, Any]:
    variant_image = _expand_padding(base_image, padding_px)
    variant_image = _pad_and_upscale(variant_image)
    filename = f"{crop_path.stem}_ocrprep_{variant_label}.jpg"
    out_path = output_dir / filename
    variant_image.save(out_path, format="JPEG", quality=95)

    merged = dict(crop)
    merged["source_crop_path"] = str(crop_path)
    merged["crop_path"] = str(out_path)
    merged["ocr_preprocess"] = preprocess_mode
    merged["ocr_preprocess_size"] = [variant_image.width, variant_image.height]
    merged["ocr_debug_group"] = _safe_group_name(crop, crop_path)
    merged["ocr_debug_variant"] = variant_label
    merged["ocr_debug_padding_px"] = padding_px
    merged["ocr_small_crop_enhanced"] = small_crop_enhanced
    return merged


def _prepare_one(
    crop: dict[str, Any],
    output_dir: Path,
    *,
    debug_enabled: bool,
    debug_variant_count: int,
    debug_pad_step_px: int,
    debug_dir: Path | None,
) -> list[dict[str, Any]]:
    base_image, preprocess_mode, crop_path, source_size = _prepare_base_image(crop)
    base_image, small_crop_enhanced = _enhance_small_crop(
        base_image,
        source_size=source_size,
    )
    if small_crop_enhanced:
        preprocess_mode = f"{preprocess_mode}+x2_smooth"
    if debug_enabled:
        group_dir = (debug_dir or output_dir) / _safe_group_name(crop, crop_path)
        group_dir.mkdir(parents=True, exist_ok=True)
        variants: list[dict[str, Any]] = [
            _build_variant_item(
                crop,
                base_image=base_image,
                preprocess_mode=preprocess_mode,
                crop_path=crop_path,
                output_dir=group_dir,
                variant_label="base",
                padding_px=0,
                small_crop_enhanced=small_crop_enhanced,
            )
        ]
        for idx in range(1, max(0, debug_variant_count) + 1):
            padding_px = max(0, debug_pad_step_px) * idx
            variants.append(
                _build_variant_item(
                    crop,
                    base_image=base_image,
                    preprocess_mode=preprocess_mode,
                    crop_path=crop_path,
                    output_dir=group_dir,
                    variant_label=f"pad_{padding_px:02d}",
                    padding_px=padding_px,
                    small_crop_enhanced=small_crop_enhanced,
                )
            )
        return variants

    return [
        _build_variant_item(
            crop,
            base_image=base_image,
            preprocess_mode=preprocess_mode,
            crop_path=crop_path,
            output_dir=output_dir,
            variant_label="base",
            padding_px=0,
            small_crop_enhanced=small_crop_enhanced,
        )
    ]


def prepare_ocr_crops(
    crops: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    debug_enabled: bool = False,
    debug_variant_count: int = 3,
    debug_pad_step_px: int = 3,
    debug_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_root = (
        Path(debug_dir).expanduser().resolve()
        if debug_enabled and debug_dir is not None
        else None
    )
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    for crop in crops:
        try:
            prepared.extend(
                _prepare_one(
                    crop,
                    out_dir,
                    debug_enabled=debug_enabled,
                    debug_variant_count=debug_variant_count,
                    debug_pad_step_px=debug_pad_step_px,
                    debug_dir=debug_root,
                )
            )
        except Exception as e:
            fallback = dict(crop)
            fallback["ocr_preprocess"] = "fallback_original"
            fallback["ocr_preprocess_error"] = repr(e)
            prepared.append(fallback)
    return prepared
