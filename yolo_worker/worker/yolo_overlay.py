# yolo_worker/yolo_overlay.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any, List
import math, hashlib
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Det:
    det_index: int
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]
    geometry: Optional[Dict[str, Any]] = None  # GeoJSON geometry


def _color_for_class(class_id: int):
    h = hashlib.md5(str(class_id).encode()).hexdigest()
    r = max(int(h[0:2], 16), 64)
    g = max(int(h[2:4], 16), 64)
    b = max(int(h[4:6], 16), 64)
    return (r, g, b)


def _safe_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _clamp_bbox(b, w: int, h: int):
    x1, y1, x2, y2 = b
    x1 = int(max(0, min(w - 1, math.floor(x1))))
    y1 = int(max(0, min(h - 1, math.floor(y1))))
    x2 = int(max(0, min(w - 1, math.ceil(x2))))
    y2 = int(max(0, min(h - 1, math.ceil(y2))))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iter_polys(geom: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    t = geom.get("type")
    c = geom.get("coordinates")
    out: List[List[Tuple[float, float]]] = []
    if not t or c is None:
        return out
    if t == "Polygon":
        outer = c[0] if c else []
        out.append([(float(x), float(y)) for x, y in outer])
    elif t == "MultiPolygon":
        for poly in c:
            outer = poly[0] if poly else []
            out.append([(float(x), float(y)) for x, y in outer])
    return out


def draw_yolo_style(img: Image.Image, dets: Sequence[Det], *, box_alpha=0.22, mask_alpha=0.18, lw=3, font_size=16):
    base = img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _safe_font(font_size)

    for det in dets:
        color = _color_for_class(det.class_id)
        x1, y1, x2, y2 = _clamp_bbox(det.bbox_xyxy, w, h)

        fill = (*color, int(255 * box_alpha))
        stroke = (*color, 255)
        d.rectangle([x1, y1, x2, y2], fill=fill, outline=stroke, width=lw)

        if det.geometry:
            for ring in _iter_polys(det.geometry):
                if len(ring) >= 3:
                    d.polygon(ring, fill=(*color, int(255 * mask_alpha)), outline=stroke)

        label = f"{det.class_name} {det.score:.2f} ({x1},{y1})-({x2},{y2})"
        try:
            tb = d.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = (len(label) * font_size // 2, font_size + 6)

        pad = 4
        lx1, ly1 = x1, max(0, y1 - (th + pad * 2))
        lx2, ly2 = min(w, lx1 + tw + pad * 2), min(h, ly1 + th + pad * 2)
        d.rectangle([lx1, ly1, lx2, ly2], fill=(*color, int(255 * 0.60)))
        d.text((lx1 + pad, ly1 + pad), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")

