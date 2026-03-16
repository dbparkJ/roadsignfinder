import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    fm = None
    plt = None
    HAS_MATPLOTLIB = False


def _configure_font() -> None:
    if not HAS_MATPLOTLIB:
        return
    if FONT_PATH:
        try:
            fm.fontManager.addfont(FONT_PATH)
            font_name = fm.FontProperties(fname=FONT_PATH).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            pass
    preferred = [
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _find_font_path() -> str | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)

    font_roots = [Path("/usr/share/fonts"), Path.home() / ".fonts"]
    keywords = ("NotoSansCJK", "NanumGothic", "DejaVuSans")
    for root in font_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                continue
            if any(keyword.lower() in path.name.lower() for keyword in keywords):
                return str(path)
    return None


FONT_PATH = _find_font_path()


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _color_map(value: float, palette: str) -> tuple[int, int, int]:
    if math.isnan(value):
        return (240, 240, 240)

    if palette == "viridis":
        stops = [
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ]
    else:
        stops = [
            (255, 255, 204),
            (254, 217, 118),
            (253, 141, 60),
            (240, 59, 32),
            (189, 0, 38),
        ]

    value = max(0.0, min(1.0, value))
    scaled = value * (len(stops) - 1)
    index = min(len(stops) - 2, int(math.floor(scaled)))
    frac = scaled - index
    return _lerp_color(stops[index], stops[index + 1], frac)


def _draw_basic_frame(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> tuple[int, int, int, int]:
    font_title = _get_font(28)
    font_label = _get_font(20)
    font_tick = _get_font(16)

    left = 110
    top = 90
    right = width - 80
    bottom = height - 110

    draw.rectangle((left, top, right, bottom), outline=(30, 30, 30), width=2)
    draw.text((left, 28), title, fill=(20, 20, 20), font=font_title)
    xw, _ = _text_size(draw, xlabel, font_label)
    draw.text(((left + right - xw) // 2, height - 55), xlabel, fill=(20, 20, 20), font=font_label)
    draw.text((16, 56), ylabel, fill=(20, 20, 20), font=font_label)

    for i in range(6):
        x = left + (right - left) * i / 5.0
        y = bottom - (bottom - top) * i / 5.0
        draw.line((x, bottom, x, bottom + 6), fill=(50, 50, 50), width=1)
        draw.line((left - 6, y, left, y), fill=(50, 50, 50), width=1)

    return left, top, right, bottom


def _draw_axis_tick_labels(
    draw: ImageDraw.ImageDraw,
    plot_box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    font_tick = _get_font(16)
    left, top, right, bottom = plot_box
    for i in range(6):
        xt = x_range[0] + (x_range[1] - x_range[0]) * i / 5.0
        yt = y_range[0] + (y_range[1] - y_range[0]) * i / 5.0
        xs = f"{xt:.0f}"
        ys = f"{yt:.0f}"
        xw, xh = _text_size(draw, xs, font_tick)
        yw, yh = _text_size(draw, ys, font_tick)
        x = left + (right - left) * i / 5.0
        y = bottom - (bottom - top) * i / 5.0
        draw.text((x - xw / 2, bottom + 12), xs, fill=(40, 40, 40), font=font_tick)
        draw.text((left - 14 - yw, y - yh / 2), ys, fill=(40, 40, 40), font=font_tick)


def _draw_vertical_color_legend(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    top: int,
    height: int,
    label: str,
    scale_min: float,
    scale_max: float,
    palette: str,
) -> None:
    font_label = _get_font(18)
    font_tick = _get_font(14)
    width = 24
    for i in range(height):
        t = 1.0 - (i / max(1, height - 1))
        color = _color_map(t, palette)
        draw.line((x, top + i, x + width, top + i), fill=color, width=1)
    draw.rectangle((x, top, x + width, top + height), outline=(50, 50, 50), width=1)
    draw.text((x - 4, top - 28), label, fill=(20, 20, 20), font=font_label)
    draw.text((x + width + 10, top - 8), f"{scale_max:.2f}", fill=(40, 40, 40), font=font_tick)
    draw.text((x + width + 10, top + height - 8), f"{scale_min:.2f}", fill=(40, 40, 40), font=font_tick)


def _save_pil_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OCR JSON result distributions with heatmaps and charts.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="ocr_test/images_ocr_json",
        help="Directory containing OCR JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_test/visualizations",
        help="Directory to save generated plots and summary files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Bin count for 2D heatmaps.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-N text and numeric tokens to plot.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return data


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _category_for(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def _collect_records(input_dir: Path) -> tuple[list[dict], Counter, Counter]:
    records: list[dict] = []
    text_counter: Counter[str] = Counter()
    numeric_counter: Counter[str] = Counter()

    for path in sorted(input_dir.rglob("*.json")):
        data = _load_json(path)
        width = data.get("width")
        height = data.get("height")
        parsing_res_list = data.get("parsing_res_list") or []
        if not isinstance(parsing_res_list, list):
            parsing_res_list = []

        texts: list[str] = []
        for item in parsing_res_list:
            if not isinstance(item, dict):
                continue
            text = str(item.get("block_content", "")).strip()
            if text:
                texts.append(text)
                text_counter[text] += 1
                for token in re.findall(r"\d+(?:\.\d+)?", text):
                    numeric_counter[token] += 1

        width_value = int(width) if isinstance(width, (int, float)) and width is not None else None
        height_value = int(height) if isinstance(height, (int, float)) and height is not None else None
        area = (
            int(width_value * height_value)
            if width_value is not None and height_value is not None
            else None
        )

        records.append(
            {
                "path": str(path),
                "category": _category_for(path, input_dir),
                "width": width_value,
                "height": height_value,
                "area": area,
                "block_count": len(parsing_res_list),
                "has_detection": int(bool(texts)),
                "text_count": len(texts),
                "char_count": sum(len(text.replace("\n", "")) for text in texts),
                "texts_joined": " | ".join(texts),
            }
        )

    return records, text_counter, numeric_counter


def _write_summary_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "category",
        "width",
        "height",
        "area",
        "block_count",
        "has_detection",
        "text_count",
        "char_count",
        "texts_joined",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _serialize_record(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {
        "path": record["path"],
        "category": record["category"],
        "width": record["width"],
        "height": record["height"],
        "area": record["area"],
        "block_count": record["block_count"],
        "text_count": record["text_count"],
        "char_count": record["char_count"],
    }


def _pick_extreme_record(
    records: list[dict],
    *,
    has_detection: int,
    mode: str,
) -> dict | None:
    candidates = [
        record
        for record in records
        if record["area"] is not None and record["has_detection"] == has_detection
    ]
    if not candidates:
        return None
    if mode == "min":
        return min(candidates, key=lambda record: (record["area"], record["width"], record["height"]))
    if mode == "max":
        return max(candidates, key=lambda record: (record["area"], record["width"], record["height"]))
    raise ValueError(f"Unsupported mode: {mode}")


def _write_overview_json(
    records: list[dict],
    text_counter: Counter[str],
    numeric_counter: Counter[str],
    output_path: Path,
) -> None:
    by_category = {}
    for category in sorted({record["category"] for record in records}):
        subset = [r for r in records if r["category"] == category]
        by_category[category] = {
            "count": len(subset),
            "detection_rate": _safe_ratio(sum(r["has_detection"] for r in subset), len(subset)),
            "avg_block_count": _safe_ratio(sum(r["block_count"] for r in subset), len(subset)),
            "avg_width": _safe_ratio(
                sum(r["width"] for r in subset if r["width"] is not None),
                sum(1 for r in subset if r["width"] is not None),
            ),
            "avg_height": _safe_ratio(
                sum(r["height"] for r in subset if r["height"] is not None),
                sum(1 for r in subset if r["height"] is not None),
            ),
            "min_detected_pixel_size": _serialize_record(
                _pick_extreme_record(subset, has_detection=1, mode="min")
            ),
            "max_undetected_pixel_size": _serialize_record(
                _pick_extreme_record(subset, has_detection=0, mode="max")
            ),
        }

    payload = {
        "total_files": len(records),
        "files_with_detection": sum(record["has_detection"] for record in records),
        "detection_rate": _safe_ratio(
            sum(record["has_detection"] for record in records),
            len(records),
        ),
        "min_detected_pixel_size": _serialize_record(
            _pick_extreme_record(records, has_detection=1, mode="min")
        ),
        "max_undetected_pixel_size": _serialize_record(
            _pick_extreme_record(records, has_detection=0, mode="max")
        ),
        "top_texts": text_counter.most_common(30),
        "top_numeric_tokens": numeric_counter.most_common(30),
        "by_category": by_category,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _plot_count_heatmap(records: list[dict], output_path: Path, bins: int) -> None:
    valid = [r for r in records if r["width"] is not None and r["height"] is not None]
    widths = np.array([r["width"] for r in valid], dtype=float)
    heights = np.array([r["height"] for r in valid], dtype=float)

    hist, xedges, yedges = np.histogram2d(widths, heights, bins=bins)
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="YlOrRd",
        )
        ax.set_title("OCR JSON Count Heatmap by Image Size")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(im, ax=ax, label="File Count")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    plot_box = _draw_basic_frame(draw, 1200, 900, "OCR JSON Count Heatmap by Image Size", "Width", "Height")
    left, top, right, bottom = plot_box
    cols, rows = hist.shape
    max_value = float(np.nanmax(hist)) if hist.size else 0.0
    for xi in range(cols):
        for yi in range(rows):
            value = hist[xi, yi]
            norm = 0.0 if max_value <= 0 else value / max_value
            color = _color_map(norm, "heat")
            x0 = left + (right - left) * xi / cols
            x1 = left + (right - left) * (xi + 1) / cols
            y0 = bottom - (bottom - top) * (yi + 1) / rows
            y1 = bottom - (bottom - top) * yi / rows
            draw.rectangle((x0, y0, x1, y1), fill=color)
    _draw_axis_tick_labels(draw, plot_box, (xedges[0], xedges[-1]), (yedges[0], yedges[-1]))
    _draw_vertical_color_legend(image, draw, right + 22, top + 20, bottom - top - 40, "Count", 0.0, max_value, "heat")
    _save_pil_image(image, output_path)


def _plot_detection_rate_heatmap(records: list[dict], output_path: Path, bins: int) -> None:
    valid = [r for r in records if r["width"] is not None and r["height"] is not None]
    widths = np.array([r["width"] for r in valid], dtype=float)
    heights = np.array([r["height"] for r in valid], dtype=float)
    detected = np.array([r["has_detection"] for r in valid], dtype=float)

    total_hist, xedges, yedges = np.histogram2d(widths, heights, bins=bins)
    detected_hist, _, _ = np.histogram2d(widths, heights, bins=[xedges, yedges], weights=detected)

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = detected_hist / total_hist
    rate[total_hist == 0] = np.nan

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            rate.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title("OCR Detection Rate Heatmap by Image Size")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(im, ax=ax, label="Detection Rate")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    plot_box = _draw_basic_frame(draw, 1200, 900, "OCR Detection Rate Heatmap by Image Size", "Width", "Height")
    left, top, right, bottom = plot_box
    cols, rows = rate.shape
    for xi in range(cols):
        for yi in range(rows):
            value = rate[xi, yi]
            color = _color_map(0.0 if math.isnan(value) else value, "viridis")
            x0 = left + (right - left) * xi / cols
            x1 = left + (right - left) * (xi + 1) / cols
            y0 = bottom - (bottom - top) * (yi + 1) / rows
            y1 = bottom - (bottom - top) * yi / rows
            draw.rectangle((x0, y0, x1, y1), fill=color)
    _draw_axis_tick_labels(draw, plot_box, (xedges[0], xedges[-1]), (yedges[0], yedges[-1]))
    _draw_vertical_color_legend(image, draw, right + 22, top + 20, bottom - top - 40, "Rate", 0.0, 1.0, "viridis")
    _save_pil_image(image, output_path)


def _plot_area_vs_blocks(records: list[dict], output_path: Path) -> None:
    valid = [r for r in records if r["area"] is not None]
    x = np.array([r["area"] for r in valid], dtype=float)
    y = np.array([r["block_count"] for r in valid], dtype=float)
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(x, y, gridsize=35, cmap="plasma", mincnt=1, xscale="log")
        ax.set_title("Image Area vs OCR Block Count")
        ax.set_xlabel("Image Area (log scale)")
        ax.set_ylabel("Detected Block Count")
        fig.colorbar(hb, ax=ax, label="Files per Bin")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    x_log = np.log10(np.maximum(x, 1.0))
    y_max = float(np.max(y)) if len(y) else 1.0
    image = Image.new("RGB", (1200, 820), "white")
    draw = ImageDraw.Draw(image)
    plot_box = _draw_basic_frame(draw, 1200, 820, "Image Area vs OCR Block Count", "log10(Image Area)", "Block Count")
    left, top, right, bottom = plot_box
    x_min = float(np.min(x_log))
    x_max = float(np.max(x_log))
    y_min = 0.0
    y_top = max(1.0, y_max)

    for record, x_value, y_value in zip(valid, x_log, y):
        px = left + (right - left) * _safe_ratio(x_value - x_min, x_max - x_min if x_max > x_min else 1.0)
        py = bottom - (bottom - top) * _safe_ratio(y_value - y_min, y_top - y_min if y_top > y_min else 1.0)
        color = (200, 70, 50) if record["has_detection"] else (70, 110, 170)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)

    _draw_axis_tick_labels(draw, plot_box, (x_min, x_max), (y_min, y_top))
    font_legend = _get_font(18)
    draw.text((right - 180, top - 36), "Blue=no text, Red=detected", fill=(30, 30, 30), font=font_legend)
    _save_pil_image(image, output_path)


def _plot_top_counter(counter: Counter[str], title: str, output_path: Path, top_k: int) -> None:
    items = counter.most_common(top_k)
    labels = [label.replace("\n", " / ") for label, _ in items]
    values = [value for _, value in items]
    if HAS_MATPLOTLIB:
        fig_height = max(6, min(12, 0.4 * len(items) + 2))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        y = np.arange(len(items))
        ax.barh(y, values, color="#2f6db0")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Frequency")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    item_count = max(1, len(items))
    image = Image.new("RGB", (1500, 90 + item_count * 42), "white")
    draw = ImageDraw.Draw(image)
    font_title = _get_font(28)
    font_label = _get_font(18)
    font_value = _get_font(17)
    draw.text((40, 24), title, fill=(20, 20, 20), font=font_title)

    bar_left = 580
    bar_right = 1400
    max_value = max(values) if values else 1

    for idx, (label, value) in enumerate(zip(labels, values)):
        y = 80 + idx * 42
        clipped = label[:48] + "..." if len(label) > 48 else label
        draw.text((30, y), clipped, fill=(30, 30, 30), font=font_label)
        width = 0 if max_value <= 0 else int((bar_right - bar_left) * (value / max_value))
        draw.rectangle((bar_left, y + 4, bar_left + width, y + 28), fill=(47, 109, 176))
        draw.rectangle((bar_left, y + 4, bar_right, y + 28), outline=(190, 190, 190), width=1)
        draw.text((bar_left + width + 8, y + 2), str(value), fill=(50, 50, 50), font=font_value)

    _save_pil_image(image, output_path)


def _plot_detection_rate_by_category(records: list[dict], output_path: Path) -> None:
    categories = sorted({record["category"] for record in records})
    rates = []
    counts = []
    for category in categories:
        subset = [r for r in records if r["category"] == category]
        counts.append(len(subset))
        rates.append(_safe_ratio(sum(r["has_detection"] for r in subset), len(subset)))

    if HAS_MATPLOTLIB:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        ax1.bar(x, counts, color="#c4d6ed", label="File Count")
        ax1.set_ylabel("File Count")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)

        ax2 = ax1.twinx()
        ax2.plot(x, rates, color="#c84c31", marker="o", linewidth=2, label="Detection Rate")
        ax2.set_ylabel("Detection Rate")
        ax2.set_ylim(0.0, 1.0)

        ax1.set_title("OCR Detection Rate by Category")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    plot_box = _draw_basic_frame(draw, 1200, 800, "OCR Detection Rate by Category", "Category", "Value")
    left, top, right, bottom = plot_box
    width = right - left
    bar_area_width = width / max(1, len(categories))
    max_count = max(counts) if counts else 1
    font_label = _get_font(18)
    font_small = _get_font(16)

    for idx, (category, count, rate) in enumerate(zip(categories, counts, rates)):
        cx = left + bar_area_width * idx + bar_area_width / 2
        bar_w = bar_area_width * 0.36
        bar_h = (bottom - top) * _safe_ratio(count, max_count)
        draw.rectangle((cx - bar_w, bottom - bar_h, cx, bottom), fill=(196, 214, 237))

        line_y = bottom - (bottom - top) * rate
        draw.ellipse((cx + 12 - 4, line_y - 4, cx + 12 + 4, line_y + 4), fill=(200, 76, 49))
        if idx > 0:
            prev_cx = left + bar_area_width * (idx - 1) + bar_area_width / 2 + 12
            prev_rate = rates[idx - 1]
            prev_y = bottom - (bottom - top) * prev_rate
            draw.line((prev_cx, prev_y, cx + 12, line_y), fill=(200, 76, 49), width=3)

        tw, _ = _text_size(draw, category, font_label)
        draw.text((cx - tw / 2, bottom + 12), category, fill=(30, 30, 30), font=font_label)
        draw.text((cx - bar_w, bottom - bar_h - 20), str(count), fill=(60, 60, 60), font=font_small)
        draw.text((cx + 18, line_y - 10), f"{rate:.2f}", fill=(120, 40, 30), font=font_small)

    _draw_axis_tick_labels(draw, plot_box, (0.0, float(len(categories) - 1)), (0.0, float(max(max_count, 1))))
    draw.text((right - 250, top - 36), "Blue bars=file count, red line=detection rate", fill=(30, 30, 30), font=font_label)
    _save_pil_image(image, output_path)


def main() -> int:
    _configure_font()
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    records, text_counter, numeric_counter = _collect_records(input_dir)
    if not records:
        raise SystemExit(f"No JSON files found under: {input_dir}")

    _write_summary_csv(records, output_dir / "ocr_result_summary.csv")
    _write_overview_json(
        records,
        text_counter,
        numeric_counter,
        output_dir / "ocr_result_overview.json",
    )

    _plot_count_heatmap(records, output_dir / "size_count_heatmap.png", bins=args.bins)
    _plot_detection_rate_heatmap(records, output_dir / "size_detection_rate_heatmap.png", bins=args.bins)
    _plot_area_vs_blocks(records, output_dir / "area_vs_block_count_hexbin.png")
    _plot_top_counter(text_counter, "Top OCR Texts", output_dir / "top_ocr_texts.png", top_k=args.top_k)
    _plot_top_counter(
        numeric_counter,
        "Top Numeric Tokens in OCR Text",
        output_dir / "top_numeric_tokens.png",
        top_k=args.top_k,
    )
    _plot_detection_rate_by_category(records, output_dir / "detection_rate_by_category.png")

    print(f"[viz] input_dir={input_dir}")
    print(f"[viz] output_dir={output_dir}")
    print(f"[viz] files={len(records)} detected={sum(r['has_detection'] for r in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
