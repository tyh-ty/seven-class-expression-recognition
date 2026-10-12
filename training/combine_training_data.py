from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path


CLASSES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine extra real-world images with the FERPlus training split"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-repeat", type=int, default=2)
    parser.add_argument(
        "--include-extra-classes",
        nargs="+",
        choices=CLASSES,
        default=list(CLASSES),
        help="Only add web images for these classes; FERPlus still keeps all classes.",
    )
    return parser.parse_args()


def images(path: Path) -> list[Path]:
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_base(base: Path) -> None:
    for split in ("train", "val", "test"):
        for label in CLASSES:
            path = base / split / label
            if not path.is_dir():
                raise FileNotFoundError(f"Missing base class directory: {path}")


def main() -> None:
    args = parse_args()
    if args.extra_repeat < 1:
        raise ValueError("--extra-repeat must be at least 1")
    validate_base(args.base)
    if not args.extra.is_dir():
        raise FileNotFoundError(f"Extra training directory not found: {args.extra}")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new output path to avoid data loss."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{args.output.name}-building-", dir=args.output.parent)
    )
    base_counts: dict[str, dict[str, int]] = {}
    extra_counts: dict[str, int] = {}
    methods: Counter[str] = Counter()
    try:
        for split in ("train", "val", "test"):
            base_counts[split] = {}
            for label in CLASSES:
                destination_root = temporary / split / label
                destination_root.mkdir(parents=True, exist_ok=True)
                base_files = images(args.base / split / label)
                base_counts[split][label] = len(base_files)
                for index, source in enumerate(base_files):
                    destination = destination_root / (
                        f"base_{index:06d}_{source.name}"
                    )
                    methods[link_or_copy(source, destination)] += 1

        for label in CLASSES:
            source_root = args.extra / label
            extra_files = (
                images(source_root)
                if label in args.include_extra_classes and source_root.is_dir()
                else []
            )
            extra_counts[label] = len(extra_files)
            destination_root = temporary / "train" / label
            for repeat in range(args.extra_repeat):
                for index, source in enumerate(extra_files):
                    destination = destination_root / (
                        f"web_r{repeat + 1}_{index:04d}_{source.name}"
                    )
                    methods[link_or_copy(source, destination)] += 1

        report = {
            "dataset": "FERPlus train plus manually curated public web images",
            "base": str(args.base),
            "extra": str(args.extra),
            "extra_repeat": args.extra_repeat,
            "included_extra_classes": args.include_extra_classes,
            "base_counts": base_counts,
            "extra_unique_counts": extra_counts,
            "combined_train_counts": {
                label: base_counts["train"][label]
                + extra_counts[label] * args.extra_repeat
                for label in CLASSES
            },
            "validation_policy": "FERPlus val is unchanged; no web images are included.",
            "test_policy": "FERPlus test is unchanged; no web images are included.",
            "file_methods": dict(methods),
        }
        (temporary / "dataset.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.output.exists():
            args.output.rmdir()
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
