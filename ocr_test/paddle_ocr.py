import gc
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .config import settings
except ImportError:
    from config import settings

os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")

_ocr_pipeline = None
_ocr_pipeline_key = None


def _debug(msg: str) -> None:
    if settings.OCR_DEBUG_LOG:
        print(msg)


def _import_paddleocr_vl():
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


def _get_image_size(image_path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        return width, height
    except Exception:
        return None, None


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
        pass


def _relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _extract_page_json_values(result: Any) -> list[Any]:
    with tempfile.TemporaryDirectory(prefix="ocr_test_page_") as tmp_dir:
        work_dir = Path(tmp_dir)
        result.save_to_json(save_path=str(work_dir))
        json_files = sorted(work_dir.glob("*.json"))
        return [_load_json(path) for path in json_files]


def _write_image_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def simplify_ocr_json_value(
    value: dict[str, Any] | list[Any] | Any,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    width = None
    height = None
    parsing_res_list: list[Any] = []

    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        parsed = value.get("parsing_res_list")
        if isinstance(parsed, list):
            parsing_res_list.extend(parsed)

    if image_path is not None and (width is None or height is None):
        fallback_width, fallback_height = _get_image_size(Path(image_path).expanduser().resolve())
        width = width if width is not None else fallback_width
        height = height if height is not None else fallback_height

    return {
        "width": width,
        "height": height,
        "parsing_res_list": parsing_res_list,
    }


def simplify_saved_ocr_payload(
    payload: dict[str, Any] | list[Any] | Any,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(payload, dict) and {"width", "height", "parsing_res_list"}.issubset(payload.keys()):
        return simplify_ocr_json_value(payload, image_path=image_path)

    json_values: list[Any] = []
    if isinstance(payload, dict):
        for page in payload.get("pages") or []:
            if isinstance(page, dict):
                json_values.extend(page.get("json_values") or [])

        if image_path is None:
            image_path = payload.get("source_path")

    if not json_values:
        return simplify_ocr_json_value({}, image_path=image_path)

    merged = simplify_ocr_json_value(json_values[0], image_path=image_path)
    for value in json_values[1:]:
        simplified = simplify_ocr_json_value(value, image_path=image_path)
        if merged["width"] is None and simplified["width"] is not None:
            merged["width"] = simplified["width"]
        if merged["height"] is None and simplified["height"] is not None:
            merged["height"] = simplified["height"]
        merged["parsing_res_list"].extend(simplified["parsing_res_list"])
    return merged


def run_ocr_on_images(
    image_paths: list[str | Path],
    input_root: str | Path,
    output_root: str | Path,
    device: str = "gpu:0",
    use_queues: bool = False,
    disable_layout: bool = False,
    disable_orientation: bool = False,
    disable_unwarp: bool = False,
) -> dict[str, Any]:
    input_root_path = Path(input_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "status": "ok",
        "engine": "PaddleOCRVL",
        "input_root": str(input_root_path),
        "output_root": str(output_root_path),
        "total": len(image_paths),
        "ok": 0,
        "fail": 0,
        "items": [],
    }

    if not image_paths:
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

    for raw_path in image_paths:
        image_path = Path(raw_path).expanduser().resolve()
        relative_path = _relative_to_root(image_path, input_root_path)
        output_path = output_root_path / relative_path.with_suffix(".json")
        _debug(f"[ocr_test] image_start path={image_path}")

        item: dict[str, Any] = {
            "source_path": str(image_path),
            "relative_path": relative_path.as_posix(),
            "output_json": str(output_path),
            "status": "ok",
            "error": None,
            "num_results": 0,
            "pages": [],
        }

        try:
            results = list(pipeline.predict(str(image_path), use_queues=use_queues))
            item["num_results"] = len(results)
            json_values: list[Any] = []
            for result in results:
                json_values.extend(_extract_page_json_values(result))

            simplified_payload = simplify_saved_ocr_payload(
                {"pages": [{"json_values": json_values}]},
                image_path=image_path,
            )

            payload["ok"] += 1
            _debug(f"[ocr_test] image_done path={image_path} results={item['num_results']}")
        except Exception as e:
            item["status"] = "fail"
            item["error"] = repr(e)
            payload["fail"] += 1
            print(f"[WARN] ocr image_failed path={image_path} error={e}")
            simplified_payload = simplify_ocr_json_value({}, image_path=image_path)

        _write_image_payload(output_path, simplified_payload)
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
