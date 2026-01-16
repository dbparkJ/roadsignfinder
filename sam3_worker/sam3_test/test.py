from transformers import Sam3Processor, Sam3Model
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
yolo = YOLO("/home/geonws/workspace/2026_project/roadsign_finder/yolo_worker/model/best.pt")

# Load image
#image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
#image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open("/home/geonws/workspace/2026_project/roadsign_finder/test_file/yaw_0/TRACK02/Camera01/Job_20250731_1053_Track12-1-1_Sphere_00087.jpg")

# YOLO inference for bboxes
yolo_results = yolo(image, verbose=False)[0]
boxes_xyxy = []
if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
    boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()

print(f"YOLO detections: {len(boxes_xyxy)}")

# Draw detections with boxes/masks and save like YOLO-style output image
image_rgb = image.convert("RGB")
overlay = image_rgb.copy()
draw = ImageDraw.Draw(overlay)

height, width = image_rgb.size[1], image_rgb.size[0]
pad = 30

def clamp_box(x1, y1, x2, y2, w, h, pad_px):
    x1p = max(0, int(x1) - pad_px)
    y1p = max(0, int(y1) - pad_px)
    x2p = min(w - 1, int(x2) + pad_px)
    y2p = min(h - 1, int(y2) + pad_px)
    return x1p, y1p, x2p, y2p

colors = []
for idx in range(len(boxes_xyxy)):
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    colors.append(color)

total_masks = 0
for idx, box in enumerate(boxes_xyxy):
    x1, y1, x2, y2 = box.tolist()
    px1, py1, px2, py2 = clamp_box(x1, y1, x2, y2, width, height, pad)
    print(f"yolo box {idx}: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) -> padded ({px1}, {py1}, {px2}, {py2})")

    crop = image_rgb.crop((px1, py1, px2, py2))
    inputs = processor(images=crop, text="pole", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"]
    boxes = results["boxes"]
    scores = results["scores"]

    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    for jdx, (mask, sbox, score) in enumerate(zip(masks, boxes, scores)):
        x1s, y1s, x2s, y2s = [int(v) for v in sbox]
        gx1, gy1, gx2, gy2 = x1s + px1, y1s + py1, x2s + px1, y2s + py1
        print(f"sam {idx}.{jdx}: box=({gx1}, {gy1}, {gx2}, {gy2}), score={float(score):.3f}")
        draw.rectangle([gx1, gy1, gx2, gy2], outline=colors[idx], width=3)

        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored_mask[mask > 0] = (*colors[idx], 90)
        mask_img = Image.fromarray(colored_mask, mode="RGBA")
        overlay_rgba = overlay.convert("RGBA")
        overlay_rgba.paste(mask_img, (px1, py1), mask_img)
        overlay = overlay_rgba.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        total_masks += 1

print(f"SAM masks found: {total_masks}")

output_path = "/home/geonws/workspace/2026_project/roadsign_finder/sam3_worker/sam3_test/output_yolo_like.png"
overlay.save(output_path)
print(f"Saved visualization to: {output_path}")

#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]
