from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


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


@dataclass(frozen=True)
class Candidate:
    path: Path
    session: str
    label: str
    perceptual_hash: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare naturally corrected webcam samples for adaptation"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/custom/feedback")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/custom/feedback-prepared")
    )
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hash-distance",
        type=int,
        default=3,
        help="Maximum 64-bit dHash distance treated as a near duplicate.",
    )
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    parser.add_argument("--min-train-per-class", type=int, default=12)
    parser.add_argument("--min-val-per-class", type=int, default=4)
    parser.add_argument("--min-sessions-per-class", type=int, default=2)
    parser.add_argument(
        "--required-labels",
        nargs="+",
        choices=CLASSES,
        default=["happiness", "neutral"],
        help="Labels that must meet all thresholds before a dataset is written.",
    )
    return parser.parse_args()


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = image.convert("L").resize(
            (9, 8), Image.Resampling.LANCZOS
        ).tobytes()
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def discover_candidates(
    root: Path, hash_distance: int
) -> tuple[list[Candidate], dict[str, object]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Feedback directory not found: {root}")

    sessions = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    ignored_sessions = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("_")
    )
    kept: list[Candidate] = []
    hashes_by_label: dict[str, list[int]] = defaultdict(list)
    duplicates: list[str] = []
    invalid: list[str] = []

    for session in sessions:
        for label in CLASSES:
            label_root = session / label
            if not label_root.is_dir():
                continue
            for path in sorted(label_root.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    perceptual_hash = difference_hash(path)
                except (OSError, UnidentifiedImageError):
                    invalid.append(str(path))
                    continue
                if any(
                    hamming_distance(perceptual_hash, previous) <= hash_distance
                    for previous in hashes_by_label[label]
                ):
                    duplicates.append(str(path))
                    continue
                hashes_by_label[label].append(perceptual_hash)
                kept.append(
                    Candidate(
                        path=path,
                        session=session.name,
                        label=label,
                        perceptual_hash=perceptual_hash,
                    )
                )

    discovery = {
        "sessions": [path.name for path in sessions],
        "ignored_sessions": ignored_sessions,
        "invalid_files": invalid,
        "near_duplicates": duplicates,
    }
    return kept, discovery


def split_sessions(
    session_names: list[str], validation_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    if len(session_names) < 2:
        raise ValueError(
            "Need natural corrections from at least two separate sessions or days "
            "before adaptation."
        )
    shuffled = session_names[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = set(shuffled[:validation_count])
    return set(shuffled[validation_count:]), validation


def class_report(
    candidates: list[Candidate],
    train_sessions: set[str],
    validation_sessions: set[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for label in CLASSES:
        class_candidates = [item for item in candidates if item.label == label]
        train_count = sum(item.session in train_sessions for item in class_candidates)
        validation_count = sum(
            item.session in validation_sessions for item in class_candidates
        )
        sessions = sorted({item.session for item in class_candidates})
        reasons: list[str] = []
        if len(class_candidates) < args.min_samples_per_class:
            reasons.append(
                f"total {len(class_candidates)} < {args.min_samples_per_class}"
            )
        if train_count < args.min_train_per_class:
            reasons.append(f"train {train_count} < {args.min_train_per_class}")
        if validation_count < args.min_val_per_class:
            reasons.append(
                f"val {validation_count} < {args.min_val_per_class}"
            )
        if len(sessions) < args.min_sessions_per_class:
            reasons.append(
                f"sessions {len(sessions)} < {args.min_sessions_per_class}"
            )
        report[label] = {
            "eligible": not reasons,
            "total": len(class_candidates),
            "train": train_count,
            "val": validation_count,
            "sessions": sessions,
            "reasons": reasons,
        }
    return report


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between zero and one")
    if not 0 <= args.hash_distance <= 64:
        raise ValueError("--hash-distance must be between 0 and 64")
    for name in (
        "min_samples_per_class",
        "min_train_per_class",
        "min_val_per_class",
        "min_sessions_per_class",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new directory to avoid data loss."
        )

    candidates, discovery = discover_candidates(args.input, args.hash_distance)
    train_sessions, validation_sessions = split_sessions(
        discovery["sessions"], args.val_ratio, args.seed
    )
    classes = class_report(
        candidates, train_sessions, validation_sessions, args
    )
    missing = [label for label in args.required_labels if not classes[label]["eligible"]]
    if missing:
        details = "; ".join(
            f"{label}: {', '.join(classes[label]['reasons'])}" for label in missing
        )
        raise ValueError(
            "Not enough independent natural corrections to train safely. " + details
        )

    eligible_labels = {
        label for label, values in classes.items() if values["eligible"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{args.output.name}-building-", dir=args.output.parent)
    )
    manifest: list[dict[str, object]] = []
    methods: dict[str, int] = defaultdict(int)
    try:
        for split in ("train", "val"):
            for label in CLASSES:
                (temporary / split / label).mkdir(parents=True, exist_ok=True)

        for item in candidates:
            if item.label not in eligible_labels:
                continue
            split = "val" if item.session in validation_sessions else "train"
            destination = (
                temporary / split / item.label / f"{item.session}__{item.path.name}"
            )
            method = link_or_copy(item.path, destination)
            methods[method] += 1
            manifest.append(
                {
                    "split": split,
                    "label": item.label,
                    "session": item.session,
                    "source": str(item.path),
                    "destination": str(destination.relative_to(temporary)),
                    "dhash64": f"{item.perceptual_hash:016x}",
                }
            )

        report = {
            "dataset": "Natural webcam correction samples",
            "input": str(args.input),
            "train_sessions": sorted(train_sessions),
            "validation_sessions": sorted(validation_sessions),
            "required_labels": args.required_labels,
            "eligible_labels": sorted(eligible_labels),
            "thresholds": {
                "hash_distance": args.hash_distance,
                "min_samples_per_class": args.min_samples_per_class,
                "min_train_per_class": args.min_train_per_class,
                "min_val_per_class": args.min_val_per_class,
                "min_sessions_per_class": args.min_sessions_per_class,
            },
            "classes": classes,
            "discovery": discovery,
            "file_methods": dict(methods),
        }
        (temporary / "dataset.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (temporary / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.output.exists():
            args.output.rmdir()
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
