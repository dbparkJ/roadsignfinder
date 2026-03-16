import argparse
from pathlib import Path
from typing import Iterable

try:
    from .config import settings
    from .paddle_ocr import release_ocr_runtime, run_ocr_on_images
except ImportError:
    from config import settings
    from paddle_ocr import release_ocr_runtime, run_ocr_on_images

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively run PaddleOCR on images and mirror results as JSON files.",
    )
    parser.add_argument("input_dir", help="Root directory containing images.")
    parser.add_argument(
        "--output-dir",
        help="Root directory for mirrored JSON outputs. Defaults to <input_dir>_ocr_json.",
    )
    parser.add_argument(
        "--device",
        default=settings.OCR_DEVICE,
        help=f"OCR device string. Default: {settings.OCR_DEVICE}",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(IMAGE_EXTENSIONS),
        help="Image file extensions to include.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output JSON already exists.",
    )
    parser.add_argument(
        "--use-queues",
        action="store_true",
        default=settings.OCR_USE_QUEUES,
        help="Pass use_queues=True to PaddleOCR predict.",
    )
    parser.add_argument(
        "--disable-layout",
        action="store_true",
        default=settings.OCR_DISABLE_LAYOUT,
        help="Disable layout detection.",
    )
    parser.add_argument(
        "--disable-orientation",
        action="store_true",
        default=settings.OCR_DISABLE_ORIENTATION,
        help="Disable document orientation classification.",
    )
    parser.add_argument(
        "--disable-unwarp",
        action="store_true",
        default=settings.OCR_DISABLE_UNWARP,
        help="Disable document unwarping.",
    )
    return parser.parse_args()


def _normalize_extensions(values: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        ext = value.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def _discover_images(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _filter_existing(images: list[Path], input_dir: Path, output_dir: Path) -> list[Path]:
    filtered: list[Path] = []
    for image_path in images:
        output_path = output_dir / image_path.relative_to(input_dir)
        if output_path.with_suffix(".json").exists():
            continue
        filtered.append(image_path)
    return filtered


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_ocr_json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = _normalize_extensions(args.extensions)
    images = _discover_images(input_dir, extensions)
    if args.skip_existing:
        images = _filter_existing(images, input_dir, output_dir)

    print(
        f"[ocr_test] input_dir={input_dir} output_dir={output_dir} "
        f"images={len(images)} device={args.device}"
    )

    if not images:
        print("[ocr_test] no matching image files found")
        return 0

    try:
        summary = run_ocr_on_images(
            image_paths=images,
            input_root=input_dir,
            output_root=output_dir,
            device=args.device,
            use_queues=args.use_queues,
            disable_layout=args.disable_layout,
            disable_orientation=args.disable_orientation,
            disable_unwarp=args.disable_unwarp,
        )
    finally:
        if settings.OCR_RELEASE_GPU_CACHE:
            release_ocr_runtime(drop_pipeline=settings.OCR_DROP_PIPELINE_AFTER_TASK)

    print(
        f"[ocr_test] status={summary['status']} "
        f"ok={summary['ok']} fail={summary['fail']} total={summary['total']}"
    )
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
