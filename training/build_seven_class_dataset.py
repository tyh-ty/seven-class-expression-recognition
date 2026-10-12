from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seven-class FERPlus training data with optional video and posed-face sources"
    )
    parser.add_argument("--base", type=Path, default=Path("data/ferplus/majority-aligned"))
    parser.add_argument("--ravdess", type=Path, default=Path("data/ravdess/faces-seven"))
    parser.add_argument("--output", type=Path, default=Path("data/seven-class/ferplus-ravdess-v1"))
    parser.add_argument("--ravdess-repeat", type=int, default=1)
    parser.add_argument("--cremad", type=Path)
    parser.add_argument("--cremad-repeat", type=int, default=1)
    parser.add_argument("--kdef", type=Path)
    parser.add_argument("--kdef-repeat", type=int, default=1)
    parser.add_argument(
        "--ravdess-frame-indices",
        type=int,
        nargs="*",
        help="Optional zero-based extracted-frame indices to include, for example: 0 1",
    )
    parser.add_argument(
        "--cremad-frame-indices",
        type=int,
        nargs="*",
        default=[1],
        help="Extracted CREMA-D frame indices used for training; defaults to the center frame.",
    )
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    if args.ravdess_repeat < 1:
        raise ValueError("--ravdess-repeat must be positive")
    if args.cremad_repeat < 1:
        raise ValueError("--cremad-repeat must be positive")
    if args.kdef_repeat < 1:
        raise ValueError("--kdef-repeat must be positive")
    if args.ravdess_frame_indices is not None and any(
        index < 0 for index in args.ravdess_frame_indices
    ):
        raise ValueError("--ravdess-frame-indices cannot contain negative values")
    if args.cremad_frame_indices is not None and any(
        index < 0 for index in args.cremad_frame_indices
    ):
        raise ValueError("--cremad-frame-indices cannot contain negative values")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    methods: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        for label in CLASSES:
            destination = args.output / split / label
            destination.mkdir(parents=True, exist_ok=True)
            base_files = sorted(
                path for path in (args.base / split / label).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not base_files:
                raise ValueError(f"Base split is empty: {split}/{label}")
            for index, source in enumerate(base_files):
                methods[link_or_copy(source, destination / f"ferplus_{index:06d}{source.suffix.lower()}")] += 1
                counts[f"{split}/{label}/ferplus"] += 1
            if split == "train":
                ravdess_files = sorted(
                    path for path in (args.ravdess / split / label).iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    and (
                        args.ravdess_frame_indices is None
                        or any(
                            path.stem.endswith(f"_f{frame_index}")
                            for frame_index in args.ravdess_frame_indices
                        )
                    )
                )
                if not ravdess_files:
                    raise ValueError(f"RAVDESS training class is empty: {label}")
                for repeat in range(args.ravdess_repeat):
                    for index, source in enumerate(ravdess_files):
                        target = destination / f"ravdess_r{repeat}_{index:06d}{source.suffix.lower()}"
                        methods[link_or_copy(source, target)] += 1
                        counts[f"{split}/{label}/ravdess"] += 1
                if args.cremad is not None and label != "surprise":
                    cremad_files = sorted(
                        path for path in (args.cremad / split / label).iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                        and (
                            args.cremad_frame_indices is None
                            or any(
                                path.stem.endswith(f"_f{frame_index}")
                                for frame_index in args.cremad_frame_indices
                            )
                        )
                    )
                    if not cremad_files:
                        raise ValueError(f"CREMA-D training class is empty: {label}")
                    for repeat in range(args.cremad_repeat):
                        for index, source in enumerate(cremad_files):
                            target = destination / f"cremad_r{repeat}_{index:06d}{source.suffix.lower()}"
                            methods[link_or_copy(source, target)] += 1
                            counts[f"{split}/{label}/cremad"] += 1
                if args.kdef is not None:
                    kdef_files = sorted(
                        path for path in (args.kdef / split / label).iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    )
                    if not kdef_files:
                        raise ValueError(f"KDEF training class is empty: {label}")
                    for repeat in range(args.kdef_repeat):
                        for index, source in enumerate(kdef_files):
                            target = destination / f"kdef_r{repeat}_{index:06d}{source.suffix.lower()}"
                            methods[link_or_copy(source, target)] += 1
                            counts[f"{split}/{label}/kdef"] += 1
    if args.cremad is not None or args.kdef is not None:
        for label in CLASSES:
            destination = args.output / "target_val" / label
            destination.mkdir(parents=True, exist_ok=True)
            target_sources = [("ravdess", args.ravdess / "val" / label)]
            if args.cremad is not None:
                target_sources.append(("cremad", args.cremad / "val" / label))
            if args.kdef is not None:
                target_sources.append(("kdef", args.kdef / "val" / label))
            for source_name, source_root in target_sources:
                source_files = sorted(
                    path for path in source_root.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                for index, source in enumerate(source_files):
                    target = destination / f"{source_name}_{index:06d}{source.suffix.lower()}"
                    methods[link_or_copy(source, target)] += 1
                    counts[f"target_val/{label}/{source_name}"] += 1
    report = {
        "classes": CLASSES,
        "base": str(args.base.resolve()),
        "ravdess": str(args.ravdess.resolve()),
        "cremad": str(args.cremad.resolve()) if args.cremad else None,
        "kdef": str(args.kdef.resolve()) if args.kdef else None,
        "validation_and_test": "unchanged FERPlus splits with contempt excluded",
        "target_validation": (
            "RAVDESS, CREMA-D, and KDEF validation identities"
            if args.cremad is not None and args.kdef is not None
            else "RAVDESS and optional domain validation identities"
            if args.cremad is not None or args.kdef is not None
            else None
        ),
        "ravdess_repeat": args.ravdess_repeat,
        "ravdess_frame_indices": args.ravdess_frame_indices,
        "cremad_repeat": args.cremad_repeat,
        "cremad_frame_indices": args.cremad_frame_indices,
        "kdef_repeat": args.kdef_repeat,
        "counts": dict(sorted(counts.items())),
        "file_methods": dict(methods),
    }
    (args.output / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
