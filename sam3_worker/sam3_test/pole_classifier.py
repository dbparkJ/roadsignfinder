from transformers import Sam3Processor, Sam3Model
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS = "/home/geonws/workspace/2026_project/roadsign_finder/yolo_worker/model/best.pt"
IMAGE_PATH = "/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK03/Camera01/Job_20250731_1053_Track13_Sphere_00354.jpg"
PAD_PX = 30
OUTPUT_PATH = "/home/geonws/workspace/2026_project/roadsign_finder/sam3_worker/sam3_test/output_pole_pipeline.png"
DEBUG = True
DEBUG_DIR = "/home/geonws/workspace/2026_project/roadsign_finder/sam3_worker/sam3_test/debug"

# --- 튜닝 파라미터 ---
HIST_SMOOTH_K = 31
HIST_PEAK_REL = 0.35          # 히스토그램 peak가 max의 몇 % 이상이면 후보로 볼지
H_BAND_HALF = 10              # peak y 주변 엣지 탐색 band(±)
H_BOX_PAD = 12                # pos_box 좌우/상하 padding
H_MIN_LEN = 40                # 가로 박스 최소 길이
H_ASPECT_MIN = 3.0            # 가로 마스크 w/h 최소
V_ASPECT_MIN = 3.5            # 세로 마스크 h/w 최소

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

def clamp_box(x1, y1, x2, y2, w, h, pad_px=0):
    x1p = max(0, int(x1) - pad_px)
    y1p = max(0, int(y1) - pad_px)
    x2p = min(w - 1, int(x2) + pad_px)
    y2p = min(h - 1, int(y2) + pad_px)
    if x2p <= x1p: x2p = min(w - 1, x1p + 1)
    if y2p <= y1p: y2p = min(h - 1, y1p + 1)
    return x1p, y1p, x2p, y2p

def bbox_from_mask(mask_bin):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def aspect_hw(mask_bin):
    b = bbox_from_mask(mask_bin)
    if b is None:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = b
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    return w, h, (h / w)

def find_horizontal_band_by_hist(crop_rgb_np, sign_mask_crop=None):
    """
    히스토그램으로 '수평 엣지'가 가장 많이 모이는 y대역(peak) 찾고,
    그 peak 주변 band에서 x-분포로 pos_box(가로폴 exemplar box)를 생성.
    """
    gray = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2GRAY)

    # (선택) 표지판 영역을 제거해서 배경/문자 엣지 영향을 줄임
    work = gray.copy()
    if sign_mask_crop is not None:
        if sign_mask_crop.shape != work.shape:
            h = min(sign_mask_crop.shape[0], work.shape[0])
            w = min(sign_mask_crop.shape[1], work.shape[1])
            sign_mask_crop = sign_mask_crop[:h, :w]
            work = work[:h, :w]
        work[sign_mask_crop > 0] = 0

    edges = cv2.Canny(work, 80, 160)

    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, 3)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0  # 0~180

    # near-horizontal: 0±15 or 180±15
    horz = (((ang < 15) | (ang > 165)) & (edges > 0)).astype(np.uint8)

    hist = horz.sum(axis=1).astype(np.float32)
    if hist.max() <= 0:
        return None, None, None  # peak_y, pos_box, hist

    # smooth
    hist_s = cv2.GaussianBlur(hist.reshape(-1, 1), (1, HIST_SMOOTH_K), 0).ravel()
    peak_y = int(np.argmax(hist_s))
    if hist_s[peak_y] < HIST_PEAK_REL * float(hist_s.max()):
        return None, None, hist_s  # peak가 약하면 포기

    H, W = horz.shape
    y1 = max(0, peak_y - H_BAND_HALF)
    y2 = min(H - 1, peak_y + H_BAND_HALF)

    ys, xs = np.where(horz[y1:y2 + 1, :] > 0)
    if len(xs) < 10:
        return peak_y, None, hist_s

    x_min = int(xs.min())
    x_max = int(xs.max())
    if (x_max - x_min) < H_MIN_LEN:
        return peak_y, None, hist_s

    # pos_box 생성 (crop 좌표)
    bx1 = max(0, x_min - H_BOX_PAD)
    bx2 = min(W - 1, x_max + H_BOX_PAD)
    by1 = max(0, peak_y - (H_BAND_HALF + H_BOX_PAD))
    by2 = min(H - 1, peak_y + (H_BAND_HALF + H_BOX_PAD))
    pos_box = [bx1, by1, bx2, by2]
    return peak_y, pos_box, hist_s

def run_sam3_text(model, processor, crop_pil, text=None, boxes=None, box_labels=None):
    """
    SAM3 PCS 호출: text / input_boxes 조합 가능.
    - box_labels: 1=pos, 0=neg
    """
    kwargs = {"images": crop_pil, "return_tensors": "pt"}
    if text is not None:
        kwargs["text"] = text
    if boxes is not None:
        # HF 문서: input_boxes shape [batch, num_boxes, 4] :contentReference[oaicite:1]{index=1}
        kwargs["input_boxes"] = [boxes]           # batch=1
        kwargs["input_boxes_labels"] = [box_labels]
    inputs = processor(**kwargs).to(DEVICE)

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
    scores = results.get("scores", None)
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    return masks, scores

def union_masks(masks):
    if masks is None or len(masks) == 0:
        return None
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    return out

def select_vertical_masks(masks):
    sel = []
    for m in masks:
        mb = (m > 0).astype(np.uint8)
        w, h, ratio = aspect_hw(mb)
        # 세로: h/w가 큼
        if (w > 0 and h > 0) and (ratio >= V_ASPECT_MIN):
            sel.append(mb)
    return sel

def select_horizontal_masks(masks, peak_y=None):
    sel = []
    for m in masks:
        mb = (m > 0).astype(np.uint8)
        w, h, ratio = aspect_hw(mb)
        # 가로: w/h가 큼 -> ratio(h/w)가 작음
        if w <= 0 or h <= 0:
            continue
        if (w / max(1, h)) < H_ASPECT_MIN:
            continue
        if peak_y is not None:
            # 마스크 중심 y가 peak 근처면 가산
            ys, xs = np.where(mb > 0)
            if len(ys) == 0:
                continue
            my = float(ys.mean())
            if abs(my - peak_y) > (H_BAND_HALF * 3):
                continue
        sel.append(mb)
    return sel

def count_vertical_poles(vert_union):
    """
    아주 간단한 '세로 기둥 개수' 추정:
    - x-프로젝션(hist)에서 peak 개수로 1/2 판단
    """
    if vert_union is None:
        return 0
    col = vert_union.sum(axis=0).astype(np.float32)
    if col.max() <= 0:
        return 0
    col = cv2.GaussianBlur(col.reshape(1, -1), (1, 31), 0).ravel()
    col /= (col.max() + 1e-6)
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

def classify_pole_shape_v2(sign_bbox_crop, vert_union, horz_union):
    """
    도로대장형식: 아주 1차 룰(튜닝 필요)
    """
    sx1, sy1, sx2, sy2 = sign_bbox_crop
    sign_w = max(1, sx2 - sx1 + 1)
    sign_cx = 0.5 * (sx1 + sx2)

    vcnt = count_vertical_poles(vert_union)
    has_horz = horz_union is not None and horz_union.sum() > 200

    # If vertical pole clearly extends above the sign, treat as attached type.
    if vert_union is not None and vert_union.sum() > 0:
        ys, _ = np.where(vert_union > 0)
        if len(ys) > 0:
            vert_top = int(ys.min())
            if vert_top < (sy1 - 5):
                return 1408

    # 오프셋(측주/내민 판정에 사용)
    offset = 0.0
    if vert_union is not None and vert_union.sum() > 0:
        ys, xs = np.where(vert_union > 0)
        pole_cx = float(xs.mean())
        offset = abs(pole_cx - sign_cx) / float(sign_w)

    # 가로보 위치(상부/암)
    horz_is_top = False
    if has_horz:
        b = bbox_from_mask(horz_union)
        if b is not None:
            hx1, hy1, hx2, hy2 = b
            # 가로보 중심이 표지판 상단보다 위에 있으면 상부보(문형 가능)
            horz_is_top = (0.5 * (hy1 + hy2)) < (sy1 - 5)

    # 분류 룰
    if (vert_union is None or vert_union.sum() < 200) and not has_horz:
        return 1408  # 부착식 후보

    if vcnt >= 2 and has_horz and horz_is_top:
        return 1406  # 문형식
    if vcnt >= 2 and not has_horz:
        return 1402  # 복주식(상부보가 잘 안보이는 케이스)

    if vcnt == 1 and has_horz:
        # 내민/편지: 오프셋이 크면 내민 쪽
        return 1407 if offset > 0.6 else 1404

    if vcnt == 1 and not has_horz:
        return 1403 if offset > 0.35 else 1401  # 측주/단주

    # 세로가 안 잡혔는데 가로만 있는 케이스(프레임밖/가림)
    if vcnt == 0 and has_horz:
        return 1406 if horz_is_top else 1499

    return 1405  # 복합/기타

def ensure_dir(path):
    if not path:
        return
    import os
    os.makedirs(path, exist_ok=True)

def save_debug_image(path, img_pil):
    if not DEBUG:
        return
    img_pil.save(path)

def mask_to_rgba(mask_bin, color_rgb, alpha=120):
    colored = np.zeros((*mask_bin.shape, 4), dtype=np.uint8)
    colored[mask_bin > 0] = (*color_rgb, alpha)
    return Image.fromarray(colored, mode="RGBA")

def main():
    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    yolo = YOLO(YOLO_WEIGHTS)

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_rgb = image.copy()
    overlay = image_rgb.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image_rgb.size
    if DEBUG:
        ensure_dir(DEBUG_DIR)
        save_debug_image(f"{DEBUG_DIR}/00_input.png", image_rgb)

    pole_masks_full, _ = run_sam3_text(
        model, processor, image_rgb,
        text="pole",
        boxes=None,
        box_labels=None
    )
    if DEBUG and pole_masks_full is not None:
        full_union = union_masks([(m > 0).astype(np.uint8) for m in pole_masks_full])
        if full_union is not None:
            dbg = image_rgb.convert("RGBA")
            m_img = mask_to_rgba(full_union, (0, 255, 255), alpha=90)
            dbg.paste(m_img, (0, 0), m_img)
            save_debug_image(f"{DEBUG_DIR}/00b_full_poles.png", dbg.convert("RGB"))

    yolo_results = yolo(image, verbose=False)[0]

    boxes_xyxy = []
    sign_masks_full = None

    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()

    # YOLO-seg 마스크가 있으면 같이 쓰기(없으면 None)
    if getattr(yolo_results, "masks", None) is not None and yolo_results.masks is not None:
        # (N, H, W) float/0~1 형태인 경우가 많음
        sign_masks_full = yolo_results.masks.data.cpu().numpy()

    print(f"YOLO detections: {len(boxes_xyxy)}")

    colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(len(boxes_xyxy))]

    total_masks = 0
    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        px1, py1, px2, py2 = clamp_box(x1, y1, x2, y2, width, height, PAD_PX)

        crop = image_rgb.crop((px1, py1, px2 + 1, py2 + 1))
        crop_np = np.array(crop)  # RGB
        if DEBUG:
            save_debug_image(f"{DEBUG_DIR}/01_crop_{idx}.png", crop)

        # crop 내부 sign bbox(좌표는 crop 기준)
        sx1, sy1, sx2, sy2 = clamp_box(x1 - px1, y1 - py1, x2 - px1, y2 - py1,
                                       crop.size[0], crop.size[1], 0)
        sign_bbox_crop = (sx1, sy1, sx2, sy2)
        if DEBUG:
            crop_dbg = crop.copy()
            d = ImageDraw.Draw(crop_dbg)
            d.rectangle([sx1, sy1, sx2, sy2], outline=(0, 255, 0), width=2)
            save_debug_image(f"{DEBUG_DIR}/02_crop_signbox_{idx}.png", crop_dbg)

        # crop 내부 sign mask(있으면)
        sign_mask_crop = None
        if sign_masks_full is not None and idx < len(sign_masks_full):
            full_m = (sign_masks_full[idx] > 0.5).astype(np.uint8)
            sign_mask_crop = full_m[py1:py2 + 1, px1:px2 + 1]
            if DEBUG:
                m_img = mask_to_rgba(sign_mask_crop, (0, 255, 0), alpha=120)
                dbg = crop.convert("RGBA")
                dbg.paste(m_img, (0, 0), m_img)
                save_debug_image(f"{DEBUG_DIR}/03_crop_signmask_{idx}.png", dbg.convert("RGB"))

        # 1) 전체 SAM3 결과에서 표지판 주변 마스크만 선별
        selected_masks = []
        for m in pole_masks_full:
            m_crop = m[py1:py2 + 1, px1:px2 + 1]
            if m_crop.sum() > 0:
                selected_masks.append(m_crop)

        vert_masks = select_vertical_masks(selected_masks)
        vert_union = union_masks(vert_masks)
        used_union = union_masks(selected_masks)
        if DEBUG and vert_union is not None:
            m_img = mask_to_rgba(vert_union, (255, 0, 0), alpha=120)
            dbg = crop.convert("RGBA")
            dbg.paste(m_img, (0, 0), m_img)
            save_debug_image(f"{DEBUG_DIR}/04_crop_vert_{idx}.png", dbg.convert("RGB"))
        if DEBUG and used_union is not None:
            m_img = mask_to_rgba(used_union, (255, 180, 0), alpha=120)
            dbg = crop.convert("RGBA")
            dbg.paste(m_img, (0, 0), m_img)
            save_debug_image(f"{DEBUG_DIR}/04b_crop_used_{idx}.png", dbg.convert("RGB"))

        horz_union = None

        # 4) 형식 분류(간단 룰)
        label_code = classify_pole_shape_v2(sign_bbox_crop, vert_union, horz_union)
        label_name = LABEL_MAP.get(label_code, LABEL_MAP[1499])
        if DEBUG:
            vcnt = count_vertical_poles(vert_union)
            has_horz = horz_union is not None and horz_union.sum() > 200
            print(f"[debug] idx={idx} vcnt={vcnt} has_horz={has_horz} label={label_code} {label_name}")

        # --- 시각화 ---
        gx1, gy1, gx2, gy2 = px1, py1, px2, py2
        draw.rectangle([gx1, gy1, gx2, gy2], outline=colors[idx], width=3)

        # sign box(참고)
        draw.rectangle([px1+sx1, py1+sy1, px1+sx2, py1+sy2], outline=(0, 255, 0), width=2)

        draw.text((gx1 + 4, gy1 + 4), f"{label_code} {label_name}", fill=colors[idx])

        # vert/horiz 마스크 overlay
        overlay_rgba = overlay.convert("RGBA")

        def paste_mask(mask_bin, color_rgb, alpha=90):
            colored = np.zeros((*mask_bin.shape, 4), dtype=np.uint8)
            colored[mask_bin > 0] = (*color_rgb, alpha)
            mask_img = Image.fromarray(colored, mode="RGBA")
            overlay_rgba.paste(mask_img, (px1, py1), mask_img)

        if vert_union is not None:
            paste_mask(vert_union, (255, 0, 0), alpha=90)   # 세로=빨강
            total_masks += 1
        if used_union is not None:
            paste_mask(used_union, (255, 180, 0), alpha=70) # 사용된 봉=주황

        overlay = overlay_rgba.convert("RGB")
        draw = ImageDraw.Draw(overlay)

    print(f"Total structural masks (vert+horz unions) overlaid: {total_masks}")
    overlay.save(OUTPUT_PATH)
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
