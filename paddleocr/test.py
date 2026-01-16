import os
import glob
import json
import shutil
from paddleocr import PaddleOCR

# =====================
# 설정
# =====================
INPUT_DIR = "/home/geonws/workspace/2026_project/roadsign_finder/paddleocr/test_file"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR 초기화
ocr = PaddleOCR(
    lang="korean",
    use_textline_orientation=True,
    device="gpu:0",    # GPU 강제 (원하면 "gpu")
)

# 지원 이미지 확장자
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# =====================
# 폴더 내 이미지 처리
# =====================
image_paths = [
    p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
    if p.lower().endswith(IMAGE_EXTS)
]

print(f"Found {len(image_paths)} images")

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]

    print(f"OCR processing: {img_name}")

    # OCR 실행
    results = ocr.predict(img_path)

    for res in results:
        # 임시 json 저장
        tmp_dir = "_tmp_json"
        os.makedirs(tmp_dir, exist_ok=True)
        res.save_to_json(tmp_dir)

        # 생성된 json 찾기
        json_files = glob.glob(os.path.join(tmp_dir, "*.json"))
        if not json_files:
            print(f"⚠️ JSON not created for {img_name}")
            continue

        tmp_json = json_files[0]
        target_json = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        # 원하는 이름으로 이동
        shutil.move(tmp_json, target_json)

        print(f"Saved: {target_json}")

        # 임시 폴더 정리
        shutil.rmtree(tmp_dir)

print("✅ All done.")

