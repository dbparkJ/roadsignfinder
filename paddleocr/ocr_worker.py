import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# paddle/paddleocr import 전에 설정해야 적용됨
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")


def _import_paddleocr_vl():
    """
    프로젝트의 로컬 `paddleocr/` 폴더가 pip 패키지 import를 가리는 문제를 피하기 위해
    임시로 sys.path에서 프로젝트 루트를 제거하고 PaddleOCRVL을 불러온다.
    """
    project_root = Path(__file__).resolve().parents[1]
    removed: list[tuple[int, str]] = []

    for i in range(len(sys.path) - 1, -1, -1):
        p = sys.path[i]
        try:
            resolved = Path(p).resolve()
        except Exception:
            resolved = None
        if p == "" or resolved == project_root:
            removed.append((i, p))
            sys.path.pop(i)

    try:
        pkg = importlib.import_module("paddleocr")
    finally:
        for i, p in sorted(removed):
            sys.path.insert(i, p)

    ocr_cls = getattr(pkg, "PaddleOCRVL", None)
    if ocr_cls is None:
        raise ImportError("Installed `paddleocr` package does not expose `PaddleOCRVL`.")
    return ocr_cls


def _load_json(path: Path) -> dict[str, Any] | list[Any] | Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def run_ocr_on_crops(
    crops: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "gpu:0",
    use_queues: bool = False,
    disable_layout: bool = False,
    disable_orientation: bool = False,
    disable_unwarp: bool = False,
) -> dict[str, Any]:
    """
    YOLO crop 이미지 목록을 받아 PaddleOCRVL 수행.
    - markdown 저장은 하지 않음
    - json 파일 저장 + 파싱 결과를 함께 반환
    """
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "status": "ok",
        "engine": "PaddleOCRVL",
        "output_dir": str(out_root),
        "total": len(crops),
        "ok": 0,
        "fail": 0,
        "items": [],
    }

    if not crops:
        return payload

    PaddleOCRVL = _import_paddleocr_vl()
    pipeline = PaddleOCRVL(
        device=device,
        use_layout_detection=not disable_layout,
        use_doc_orientation_classify=not disable_orientation,
        use_doc_unwarping=not disable_unwarp,
    )

    for crop in crops:
        det_index = crop.get("det_index")
        crop_path = Path(str(crop.get("crop_path", ""))).expanduser().resolve()

        item: dict[str, Any] = {
            "det_index": det_index,
            "crop_path": str(crop_path),
            "bbox_xyxy": crop.get("bbox_xyxy"),
            "class_id": crop.get("class_id"),
            "class_name": crop.get("class_name"),
            "confidence": crop.get("confidence"),
            "status": "ok",
            "error": None,
            "num_results": 0,
            "pages": [],
        }

        try:
            results = list(pipeline.predict(str(crop_path), use_queues=use_queues))
            item["num_results"] = len(results)

            for r_i, res in enumerate(results):
                page_dir = out_root / f"det_{int(det_index):03d}" / f"page_{r_i:03d}"
                page_dir.mkdir(parents=True, exist_ok=True)

                before = {p.resolve() for p in page_dir.glob("*.json")}
                res.save_to_json(save_path=str(page_dir))
                after = sorted(page_dir.glob("*.json"))

                json_files = [p for p in after if p.resolve() not in before]
                if not json_files:
                    json_files = after

                page_payload = {
                    "page_index": r_i,
                    "json_files": [str(p.resolve()) for p in json_files],
                    "json_values": [_load_json(p) for p in json_files],
                }
                item["pages"].append(page_payload)

            payload["ok"] += 1
        except Exception as e:
            item["status"] = "fail"
            item["error"] = repr(e)
            payload["fail"] += 1

        payload["items"].append(item)

    if payload["fail"] == 0:
        payload["status"] = "ok"
    elif payload["ok"] == 0:
        payload["status"] = "fail"
    else:
        payload["status"] = "partial"

    return payload
