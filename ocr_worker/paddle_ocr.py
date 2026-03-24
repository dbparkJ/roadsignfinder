import importlib
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import settings

# paddle/paddleocr import 전에 설정해야 적용됨
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")


def _debug(msg: str) -> None:
    if settings.OCR_DEBUG_LOG:
        print(msg)


_ocr_pipeline = None
_ocr_pipeline_key = None


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


def _safe_output_token(value: Any, fallback: str) -> str:
    token = str(value or "").strip() or fallback
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token)
    sanitized = sanitized.strip("_")
    return sanitized or fallback


def _collect_detected_texts(value: Any, texts: list[str]) -> None:
    if isinstance(value, dict):
        content = value.get("block_content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                texts.append(text)
        for nested in value.values():
            _collect_detected_texts(nested, texts)
        return

    if isinstance(value, list):
        for nested in value:
            _collect_detected_texts(nested, texts)


def _extract_page_texts(json_values: list[Any]) -> list[str]:
    texts: list[str] = []
    for value in json_values:
        if isinstance(value, dict):
            parsed = value.get("parsing_res_list")
            if isinstance(parsed, list):
                _collect_detected_texts(parsed, texts)
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _build_pipeline(
    device: str,
    disable_layout: bool,
    disable_orientation: bool,
    disable_unwarp: bool,
):
    PaddleOCRVL = _import_paddleocr_vl()
    return PaddleOCRVL(
        device=device,
        use_layout_detection=not disable_layout,
        use_doc_orientation_classify=not disable_orientation,
        use_doc_unwarping=not disable_unwarp,
    )


def _get_pipeline(
    device: str,
    disable_layout: bool,
    disable_orientation: bool,
    disable_unwarp: bool,
):
    global _ocr_pipeline, _ocr_pipeline_key
    key = (device, disable_layout, disable_orientation, disable_unwarp)
    if _ocr_pipeline is None or _ocr_pipeline_key != key:
        _ocr_pipeline = _build_pipeline(
            device=device,
            disable_layout=disable_layout,
            disable_orientation=disable_orientation,
            disable_unwarp=disable_unwarp,
        )
        _ocr_pipeline_key = key
    return _ocr_pipeline


def release_ocr_runtime(drop_pipeline: bool = False) -> None:
    """
    OCR 작업 종료 후 GPU 캐시를 비워 메모리 누적을 완화한다.
    """
    global _ocr_pipeline, _ocr_pipeline_key
    if drop_pipeline:
        try:
            _ocr_pipeline = None
            _ocr_pipeline_key = None
        except Exception:
            pass

    try:
        gc.collect()
    except Exception:
        pass

    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        # paddle 버전/환경 차이로 비우기 실패해도 작업에는 영향 없게 처리
        pass


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

    if settings.OCR_REUSE_PIPELINE:
        pipeline = _get_pipeline(
            device=device,
            disable_layout=disable_layout,
            disable_orientation=disable_orientation,
            disable_unwarp=disable_unwarp,
        )
    else:
        pipeline = _build_pipeline(
            device=device,
            disable_layout=disable_layout,
            disable_orientation=disable_orientation,
            disable_unwarp=disable_unwarp,
        )

    for crop in crops:
        det_index = crop.get("det_index")
        crop_path = Path(str(crop.get("crop_path", ""))).expanduser().resolve()
        _debug(f"[ocr] crop_start det_index={det_index} path={crop_path}")
        try:
            default_group = f"det_{int(det_index):03d}"
        except Exception:
            default_group = crop_path.stem or "det"
        output_group = _safe_output_token(crop.get("ocr_debug_group"), default_group)
        output_variant = _safe_output_token(crop.get("ocr_debug_variant"), "base")

        item: dict[str, Any] = {
            "det_index": det_index,
            "crop_path": str(crop_path),
            "source_crop_path": crop.get("source_crop_path"),
            "bbox_xyxy": crop.get("bbox_xyxy"),
            "class_id": crop.get("class_id"),
            "class_name": crop.get("class_name"),
            "confidence": crop.get("confidence"),
            "ocr_target": crop.get("ocr_target"),
            "ocr_category": crop.get("ocr_category"),
            "ocr_shape": crop.get("ocr_shape"),
            "ocr_policy_source": crop.get("ocr_policy_source"),
            "ocr_preprocess": crop.get("ocr_preprocess"),
            "ocr_preprocess_size": crop.get("ocr_preprocess_size"),
            "ocr_debug_group": crop.get("ocr_debug_group"),
            "ocr_debug_variant": crop.get("ocr_debug_variant"),
            "ocr_debug_padding_px": crop.get("ocr_debug_padding_px"),
            "status": "ok",
            "error": None,
            "num_results": 0,
            "detected_texts": [],
            "detected_text": "",
            "detected_text_count": 0,
            "pages": [],
        }

        try:
            results = list(pipeline.predict(str(crop_path), use_queues=use_queues))
            item["num_results"] = len(results)

            for r_i, res in enumerate(results):
                page_dir = out_root / output_group / output_variant / f"page_{r_i:03d}"
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
                page_payload["detected_texts"] = _extract_page_texts(page_payload["json_values"])
                item["pages"].append(page_payload)

            detected_texts: list[str] = []
            for page in item["pages"]:
                for text in page.get("detected_texts") or []:
                    if text not in detected_texts:
                        detected_texts.append(text)
            item["detected_texts"] = detected_texts
            item["detected_text"] = " | ".join(detected_texts)
            item["detected_text_count"] = len(detected_texts)
            payload["ok"] += 1
            _debug(
                f"[ocr] crop_done det_index={det_index} results={item['num_results']}"
            )
        except Exception as e:
            item["status"] = "fail"
            item["error"] = repr(e)
            payload["fail"] += 1
            print(f"[WARN] ocr crop_failed det_index={det_index} error={e}")

        payload["items"].append(item)

    if not settings.OCR_REUSE_PIPELINE:
        try:
            del pipeline
        except Exception:
            pass

    if payload["fail"] == 0:
        payload["status"] = "ok"
    elif payload["ok"] == 0:
        payload["status"] = "fail"
    else:
        payload["status"] = "partial"

    return payload
