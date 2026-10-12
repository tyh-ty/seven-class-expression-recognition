from __future__ import annotations

import argparse
import json
import os
import shutil
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
        description=(
            "Build an identity-separated low-intensity adaptation set from RAVDESS "
            "onset frames while preserving the existing seven-class product data."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/seven-class/glasses-adaptation-v1"),
    )
    parser.add_argument(
        "--ravdess", type=Path, default=Path("data/ravdess/faces-seven")
    )
    parser.add_argument(
        "--ravdess-test",
        type=Path,
        default=Path("data/ravdess/faces-seven-test-complete"),
        help="Complete recovered actor-disjoint test set (actors 21-24).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/seven-class/subtle-adaptation-v2"),
    )
    parser.add_argument(
        "--extra-repeats",
        type=int,
        default=3,
        help="Additional copies of each selected onset frame; the base already contains one.",
    )
    parser.add_argument(
        "--frame-indices", type=int, nargs="+", default=[0, 1]
    )
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def is_low_intensity_onset(path: Path, frame_indices: set[int]) -> bool:
    # actor01_01-01-05-01-01-01-01_f0.jpg: fourth RAVDESS field is intensity.
    try:
        encoded_name, frame_text = path.stem.rsplit("_f", 1)
        ravdess_name = encoded_name.split("_", 1)[1]
        fields = ravdess_name.split("-")
        return (
            len(fields) == 7
            and fields[3] == "01"
            and int(frame_text) in frame_indices
        )
    except (IndexError, ValueError):
        return False


def image_files(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.extra_repeats < 1:
        raise ValueError("--extra-repeats must be positive")
    frame_indices = set(args.frame_indices)
    if not frame_indices or min(frame_indices) < 0:
        raise ValueError("--frame-indices must contain non-negative values")

    methods: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    temporary = args.output.with_name(f".{args.output.name}-building")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite unfinished build: {temporary}")

    try:
        # Preserve the product training set and its untouched FERPlus validation/test.
        for split in ("train", "val", "test", "glasses-val"):
            for label in CLASSES:
                source_root = args.base / split / label
                destination_root = temporary / split / label
                destination_root.mkdir(parents=True, exist_ok=True)
                for source in image_files(source_root):
                    methods[link_or_copy(source, destination_root / source.name)] += 1
                    counts[f"base/{split}/{label}"] += 1

        # Actors 01-16 train the adapter. Only normal-intensity onset/early frames
        # are repeated; strong-intensity clips and later apex frames are excluded.
        for label in CLASSES:
            selected = [
                path
                for path in image_files(args.ravdess / "train" / label)
                if is_low_intensity_onset(path, frame_indices)
            ]
            if not selected:
                raise RuntimeError(f"No low-intensity RAVDESS train images for {label}")
            destination_root = temporary / "train" / label
            for repeat in range(args.extra_repeats):
                for index, source in enumerate(selected):
                    name = f"ravdess_subtle_r{repeat}_{index:05d}{source.suffix.lower()}"
                    methods[link_or_copy(source, destination_root / name)] += 1
                    counts[f"adaptation/train/{label}"] += 1

        # Actors 17-20 select a candidate; actors 21-24 remain untouched final test.
        for source_root, source_split, destination_split in (
            (args.ravdess, "val", "subtle-val"),
            (args.ravdess_test, "test", "subtle-test"),
        ):
            for label in CLASSES:
                destination_root = temporary / destination_split / label
                destination_root.mkdir(parents=True, exist_ok=True)
                selected = [
                    path
                    for path in image_files(source_root / source_split / label)
                    if is_low_intensity_onset(path, frame_indices)
                ]
                if not selected:
                    raise RuntimeError(
                        f"No low-intensity RAVDESS {source_split} images for {label}"
                    )
                for source in selected:
                    methods[link_or_copy(source, destination_root / source.name)] += 1
                    counts[f"adaptation/{destination_split}/{label}"] += 1

        report = {
            "purpose": "low-intensity/onset seven-class adaptation",
            "base": str(args.base.resolve()),
            "ravdess": str(args.ravdess.resolve()),
            "ravdess_complete_test": str(args.ravdess_test.resolve()),
            "selected_intensity": "01 normal intensity",
            "selected_frame_indices": sorted(frame_indices),
            "selected_frame_positions": [0.35, 0.45],
            "extra_training_repeats": args.extra_repeats,
            "identity_separation": {
                "train_actors": list(range(1, 17)),
                "validation_actors": list(range(17, 21)),
                "final_test_actors": list(range(21, 25)),
            },
            "excluded_from_adaptation": [
                "RAVDESS strong intensity 02",
                "later frames f2-f4",
                "CK+ peak-expression images",
            ],
            "counts": dict(sorted(counts.items())),
            "file_methods": dict(methods),
        }
        (temporary / "dataset.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
