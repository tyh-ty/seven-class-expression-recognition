from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split emotion data by subject")
    parser.add_argument("--input", type=Path, default=Path("data/custom/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/custom/split"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def images_for(subject: Path, label: str) -> list[Path]:
    folder = subject / label
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def allocate_subjects(
    subjects: list[Path], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[Path]]:
    if len(subjects) < 3:
        raise ValueError(
            "At least three subjects are required for subject-level train/val/test splitting"
        )
    shuffled = subjects[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * val_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    while validation_count + test_count >= len(shuffled):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Not enough subjects to create three non-empty splits")
    return {
        "test": shuffled[:test_count],
        "val": shuffled[test_count : test_count + validation_count],
        "train": shuffled[test_count + validation_count :],
    }


def validate_splits(splits: dict[str, list[Path]]) -> None:
    for split, subjects in splits.items():
        missing = [
            label
            for label in CLASSES
            if not any(images_for(subject, label) for subject in subjects)
        ]
        if missing:
            names = ", ".join(subject.name for subject in subjects)
            raise ValueError(
                f"Split {split} with subjects [{names}] has no images for: {missing}"
            )


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1:
        raise ValueError("Validation and test ratios must be between zero and one")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than one")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new directory to avoid data loss."
        )
    subjects = sorted(
        path for path in args.input.iterdir() if path.is_dir()
    ) if args.input.is_dir() else []
    splits = allocate_subjects(subjects, args.val_ratio, args.test_ratio, args.seed)
    validate_splits(splits)
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split, split_subjects in splits.items():
        for label in CLASSES:
            destination = args.output / split / label
            destination.mkdir(parents=True, exist_ok=True)
            copied = 0
            for subject in split_subjects:
                for source in images_for(subject, label):
                    target = destination / f"{subject.name}__{source.name}"
                    shutil.copy2(source, target)
                    copied += 1
            counts[split][label] = copied
    for split in ("train", "val", "test"):
        print(split, counts[split], "total=", sum(counts[split].values()))


if __name__ == "__main__":
    main()
