from transformers import Sam3Processor, Sam3Model
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import numpy as np
import os
import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS = "/home/geonws/workspace/2026_project/roadsign_finder/yolo_worker/model/best.pt"
IMAGE_PATH = "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK03/Camera01/Job_20250731_1053_Track13_Sphere_00354.jpg"
IMAGE_DIR = "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK03/Camera01"
OUTPUT_DIR = "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK03/output_pole_v1_2"

DEBUG = False
DEBUG_DIR = "/home/geonws/workspace/2026_project/roadsign_finder/sam3_worker/sam3_test/debug_v1_2"

# Strip sizes (no padding box expansion; use full-width/full-height strips)
V_STRIP_SCALE = 1.0   # vertical strip width multiplier vs sign width
H_STRIP_SCALE = 1.0   # horizontal strip height multiplier vs sign height
V_MAX_OFFSET_RATIO = 1.0  # vertical pole center must be within sign_w * ratio from sign center
V_MAX_EDGE_DIST_RATIO = 0.6  # max horizontal distance from sign box edge to pole pixels (sign_w * ratio)

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


def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def clamp_box(x1, y1, x2, y2, w, h):
    x1p = max(0, int(x1))
    y1p = max(0, int(y1))
    x2p = min(w - 1, int(x2))
    y2p = min(h - 1, int(y2))
    if x2p <= x1p:
        x2p = min(w - 1, x1p + 1)
    if y2p <= y1p:
        y2p = min(h - 1, y1p + 1)
    return x1p, y1p, x2p, y2p


def mask_to_rgba(mask_bin, color_rgb, alpha=120):
    colored = np.zeros((*mask_bin.shape, 4), dtype=np.uint8)
    colored[mask_bin > 0] = (*color_rgb, alpha)
    return Image.fromarray(colored, mode="RGBA")


def union_masks(masks):
    if masks is None or len(masks) == 0:
        return None
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    return out


def mask_center_x(mask_union):
    if mask_union is None:
        return None
    ys, xs = np.where(mask_union > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean())


def min_horizontal_distance_to_box(mask_union, box):
    if mask_union is None:
        return None
    ys, xs = np.where(mask_union > 0)
    if len(xs) == 0:
        return None
    sx1, _, sx2, _ = box
    dx = 0.0
    if xs.max() < sx1:
        dx = float(sx1 - xs.max())
    elif xs.min() > sx2:
        dx = float(xs.min() - sx2)
    else:
        dx = 0.0
    return dx

def count_vertical_poles(mask_union):
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


def select_horizontal_masks(masks, aspect_min=3.0):
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


def classify_pole_shape_v1_2(sign_box, vert_union, horz_union):
    sx1, sy1, sx2, sy2 = sign_box
    sign_w = max(1, sx2 - sx1 + 1)
    sign_cx = 0.5 * (sx1 + sx2)

    vcnt = count_vertical_poles(vert_union)
    has_horz = horz_union is not None and horz_union.sum() > 200

    # if vertical pole extends above sign -> attached
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


def run_sam3_text(model, processor, image_pil, text):
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
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


def list_images(path_or_dir):
    if os.path.isdir(path_or_dir):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(path_or_dir, p)))
        return sorted(files)
    return [path_or_dir]


def build_output_name(label_codes, label_names, image_path):
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    if not label_codes:
        return f"unknown_unknown_{name}{ext}"
    uniq = sorted(set(zip(label_codes, label_names)))
    if len(uniq) == 1:
        code, lname = uniq[0]
        return f"{code}_{lname}_{name}{ext}"
    return f"multi_multi_{name}{ext}"


def main():
    if DEBUG:
        ensure_dir(DEBUG_DIR)
    ensure_dir(OUTPUT_DIR)

    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    yolo = YOLO(YOLO_WEIGHTS)

    input_list = list_images(IMAGE_DIR if os.path.isdir(IMAGE_DIR) else IMAGE_PATH)

    for image_path in input_list:
        image = Image.open(image_path).convert("RGB")
        image_rgb = image.copy()
        overlay = image_rgb.copy()
        draw = ImageDraw.Draw(overlay)
        width, height = image_rgb.size

        base = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = os.path.join(DEBUG_DIR, base)
        if DEBUG:
            ensure_dir(debug_path)

        yolo_results = yolo(image, 
                            verbose=False,
                            conf=0.8)[0]
        boxes_xyxy = []
        if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()

        pole_masks_full = run_sam3_text(model, processor, image_rgb, text="pole")
        full_union = union_masks([(m > 0).astype(np.uint8) for m in pole_masks_full])
        if DEBUG and full_union is not None:
            dbg = image_rgb.convert("RGBA")
            m_img = mask_to_rgba(full_union, (0, 255, 255), alpha=90)
            dbg.paste(m_img, (0, 0), m_img)
            dbg.convert("RGB").save(f"{debug_path}/00_full_poles.png")

        colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(len(boxes_xyxy))]
        label_codes = []
        label_names = []

        for idx, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)
            sign_w = max(1, x2 - x1 + 1)
            sign_h = max(1, y2 - y1 + 1)

            v_strip_w = int(sign_w * V_STRIP_SCALE)
            h_strip_h = int(sign_h * H_STRIP_SCALE)

            vx1 = int(0.5 * (x1 + x2) - 0.5 * v_strip_w)
            vx2 = vx1 + v_strip_w
            hx1 = 0
            hx2 = width - 1

            vy1 = 0
            vy2 = height - 1
            hy1 = int(0.5 * (y1 + y2) - 0.5 * h_strip_h)
            hy2 = hy1 + h_strip_h

            vx1, vy1, vx2, vy2 = clamp_box(vx1, vy1, vx2, vy2, width, height)
            hx1, hy1, hx2, hy2 = clamp_box(hx1, hy1, hx2, hy2, width, height)

            v_strip = image_rgb.crop((vx1, vy1, vx2 + 1, vy2 + 1))
            h_strip = image_rgb.crop((hx1, hy1, hx2 + 1, hy2 + 1))

            if DEBUG:
                v_dbg = v_strip.copy()
                d = ImageDraw.Draw(v_dbg)
                d.rectangle([x1 - vx1, y1 - vy1, x2 - vx1, y2 - vy1], outline=(0, 255, 0), width=2)
                v_dbg.save(f"{debug_path}/01_vstrip_{idx}.png")
                h_dbg = h_strip.copy()
                d2 = ImageDraw.Draw(h_dbg)
                d2.rectangle([x1 - hx1, y1 - hy1, x2 - hx1, y2 - hy1], outline=(0, 255, 0), width=2)
                h_dbg.save(f"{debug_path}/02_hstrip_{idx}.png")

            v_masks = run_sam3_text(model, processor, v_strip, text="pole")
            h_masks = run_sam3_text(model, processor, h_strip, text="pole")

            v_union = union_masks([(m > 0).astype(np.uint8) for m in v_masks])
            h_sel = select_horizontal_masks(h_masks, aspect_min=3.0)
            h_union = union_masks(h_sel)

            if DEBUG and v_union is not None:
                dbg = v_strip.convert("RGBA")
                m_img = mask_to_rgba(v_union, (255, 0, 0), alpha=110)
                dbg.paste(m_img, (0, 0), m_img)
                dbg.convert("RGB").save(f"{debug_path}/03_vstrip_poles_{idx}.png")
            if DEBUG and h_union is not None:
                dbg = h_strip.convert("RGBA")
                m_img = mask_to_rgba(h_union, (0, 120, 255), alpha=110)
                dbg.paste(m_img, (0, 0), m_img)
                dbg.convert("RGB").save(f"{debug_path}/04_hstrip_poles_{idx}.png")

            # Project strip masks into full image coordinates for overlay
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
            if DEBUG and h_full is not None:
                dbg = image_rgb.convert("RGBA")
                m_img = mask_to_rgba(h_full, (0, 120, 255), alpha=90)
                dbg.paste(m_img, (0, 0), m_img)
                dbg.convert("RGB").save(f"{debug_path}/05_hfull_{idx}.png")

            # Reject far vertical poles by distance to sign center
            sign_cx = 0.5 * (x1 + x2)
            pole_cx = mask_center_x(v_full)
            if pole_cx is not None:
                max_offset = max(1.0, sign_w * V_MAX_OFFSET_RATIO)
                if abs(pole_cx - sign_cx) > max_offset:
                    if DEBUG:
                        print(f"[debug] {os.path.basename(image_path)} idx={idx} far pole rejected: "
                              f"pole_cx={pole_cx:.1f} sign_cx={sign_cx:.1f} max_off={max_offset:.1f}")
                    v_full = None

            # Reject poles too far from sign box edge (horizontal distance)
            dist_edge = min_horizontal_distance_to_box(v_full, (x1, y1, x2, y2))
            if dist_edge is not None:
                max_edge = max(1.0, sign_w * V_MAX_EDGE_DIST_RATIO)
                if dist_edge > max_edge:
                    if DEBUG:
                        print(f"[debug] {os.path.basename(image_path)} idx={idx} pole too far from sign: "
                              f"dist_edge={dist_edge:.1f} max_edge={max_edge:.1f}")
                    v_full = None

            label_code = classify_pole_shape_v1_2((x1, y1, x2, y2), v_full, h_full)
            label_name = LABEL_MAP.get(label_code, LABEL_MAP[1499])
            label_codes.append(label_code)
            label_names.append(label_name)

            print(f"{os.path.basename(image_path)} sign {idx}: code={label_code} name={label_name}")

            draw.rectangle([x1, y1, x2, y2], outline=colors[idx], width=3)
            draw.text((x1 + 4, y1 + 4), f"{label_code} {label_name}", fill=colors[idx])

            overlay_rgba = overlay.convert("RGBA")
            if v_full is not None:
                overlay_rgba.paste(mask_to_rgba(v_full, (255, 0, 0), alpha=80), (0, 0),
                                   mask_to_rgba(v_full, (255, 0, 0), alpha=80))
            if h_full is not None:
                overlay_rgba.paste(mask_to_rgba(h_full, (0, 120, 255), alpha=70), (0, 0),
                                   mask_to_rgba(h_full, (0, 120, 255), alpha=70))
            overlay = overlay_rgba.convert("RGB")
            draw = ImageDraw.Draw(overlay)

        output_name = build_output_name(label_codes, label_names, image_path)
        output_path = os.path.join(OUTPUT_DIR, output_name)
        overlay.save(output_path)
        print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
