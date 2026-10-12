from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sadness": "sadness",
    "surprise": "surprise",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SUBJECT_PATTERN = re.compile(r"^(S\d{3})_")


def images(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def link(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def split_subjects(
    subject_counts: dict[str, Counter[str]], seed: int
) -> dict[str, str]:
    subjects = sorted(subject_counts)
    totals = Counter()
    for counts in subject_counts.values():
        totals.update(counts)
    targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_sizes = {
        "train": round(len(subjects) * targets["train"]),
        "val": round(len(subjects) * targets["val"]),
    }
    rng = random.Random(seed)
    best_score = float("inf")
    best_assignment: dict[str, str] | None = None
    for _ in range(20_000):
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        train_end = split_sizes["train"]
        val_end = train_end + split_sizes["val"]
        groups = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }
        score = 0.0
        valid = True
        for split, members in groups.items():
            counts = Counter()
            for subject in members:
                counts.update(subject_counts[subject])
            for label in LABEL_MAP.values():
                if counts[label] == 0:
                    valid = False
                    break
                expected = totals[label] * targets[split]
                score += ((counts[label] - expected) / max(expected, 1.0)) ** 2
            if not valid:
                break
        if valid and score < best_score:
            best_score = score
            best_assignment = {
                subject: split for split, members in groups.items() for subject in members
            }
    if best_assignment is None:
        raise RuntimeError("Could not create a subject-disjoint split with every class represented")
    return best_assignment


def perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(np.float32(resized))[:8, :8]
    bits = transformed > np.median(transformed[1:, :])
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def duplicate_audit(
    extracted: list[tuple[Path, str, str, str]], reference: Path | list[Path]
) -> tuple[set[Path], dict[str, object]]:
    references = [reference] if isinstance(reference, Path) else reference
    reference_paths = [path for root in references for path in images(root)]
    exact_reference: dict[str, Path] = {}
    reference_hashes: list[tuple[int, Path]] = []
    seen_files: set[tuple[int, int]] = set()
    for path in reference_paths:
        stat = path.stat()
        file_key = (int(stat.st_dev), int(stat.st_ino))
        if file_key in seen_files:
            continue
        seen_files.add(file_key)
        payload = path.read_bytes()
        exact_reference.setdefault(hashlib.sha256(payload).hexdigest(), path)
        reference_hashes.append((perceptual_hash(path), path))

    rejected: set[Path] = set()
    matches: list[dict[str, object]] = []
    for path, split, label, subject in extracted:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in exact_reference:
            rejected.add(path)
            matches.append(
                {
                    "source": str(path),
                    "reference": str(exact_reference[digest]),
                    "kind": "exact",
                    "distance": 0,
                    "split": split,
                    "label": label,
                    "subject": subject,
                }
            )
            continue
        candidate_hash = perceptual_hash(path)
        distance, matched_path = min(
            ((candidate_hash ^ item_hash).bit_count(), item_path)
            for item_hash, item_path in reference_hashes
        )
        if distance <= 3:
            rejected.add(path)
            matches.append(
                {
                    "source": str(path),
                    "reference": str(matched_path),
                    "kind": "near",
                    "distance": distance,
                    "split": split,
                    "label": label,
                    "subject": subject,
                }
            )
    return rejected, {
        "references": [str(root) for root in references],
        "reference_unique_files": len(reference_hashes),
        "threshold": "64-bit pHash Hamming distance <= 3",
        "rejected": len(rejected),
        "matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a subject-disjoint CK+ adaptation dataset"
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--base", type=Path, default=Path("data/seven-class/glasses-adaptation-v1")
    )
    parser.add_argument(
        "--domain-validation",
        type=Path,
        default=Path("data/seven-class/fear-adaptation-v1/target-val"),
    )
    parser.add_argument(
        "--raw-output", type=Path, default=Path("data/ckplus/faces-subject-split-v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/seven-class/ckplus-adaptation-v1")
    )
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--anger-fear-repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for destination in (args.raw_output, args.output):
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
    if args.repeat < 1 or args.anger_fear_repeat < args.repeat:
        raise ValueError("Invalid repeat values")

    table = pq.read_table(args.parquet)
    required = {"image", "label", "file"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Parquet must contain {sorted(required)}")
    rows = table.select(["image", "label", "file"]).to_pylist()
    records: list[tuple[bytes, str, str, str]] = []
    subject_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skipped_labels = Counter()
    for row in rows:
        source_label = str(row["label"]).lower()
        label = LABEL_MAP.get(source_label)
        if label is None:
            skipped_labels[source_label] += 1
            continue
        filename = str(row["file"])
        match = SUBJECT_PATTERN.match(filename)
        if not match:
            raise ValueError(f"Missing CK+ subject id in {filename!r}")
        payload = row["image"]["bytes"]
        if not payload:
            raise ValueError(f"Missing image bytes for {filename}")
        subject = match.group(1)
        records.append((payload, label, filename, subject))
        subject_counts[subject][label] += 1

    assignment = split_subjects(subject_counts, args.seed)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_temporary = Path(
        tempfile.mkdtemp(prefix="ckplus-raw-", dir=args.raw_output.parent)
    )
    adaptation_temporary = Path(
        tempfile.mkdtemp(prefix="ckplus-adaptation-", dir=args.output.parent)
    )
    try:
        for split in ("train", "val", "test"):
            for label in CLASSES:
                (raw_temporary / split / label).mkdir(parents=True)
        extracted: list[tuple[Path, str, str, str]] = []
        for payload, label, filename, subject in records:
            split = assignment[subject]
            destination = raw_temporary / split / label / f"{filename}.png"
            destination.write_bytes(payload)
            extracted.append((destination, split, label, subject))

        rejected, audit = duplicate_audit(extracted, args.base)
        for path in rejected:
            path.unlink()

        counts: Counter[str] = Counter()
        methods: Counter[str] = Counter()
        for split in ("train", "val", "test"):
            for label in CLASSES:
                destination = adaptation_temporary / split / label
                destination.mkdir(parents=True)
                for source in images(args.base / split / label):
                    methods[link(source, destination / source.name)] += 1
                    counts[f"base/{split}/{label}"] += 1

        for label in CLASSES:
            train_destination = adaptation_temporary / "train" / label
            repetitions = args.anger_fear_repeat if label in {"anger", "fear"} else args.repeat
            for repeat in range(repetitions):
                for source in images(raw_temporary / "train" / label):
                    target = train_destination / f"ckplus_r{repeat:02d}_{source.name}"
                    methods[link(source, target)] += 1
                    counts[f"ckplus/train/{label}"] += 1

        for label in CLASSES:
            target_destination = adaptation_temporary / "target-val" / label
            target_destination.mkdir(parents=True)
            for index, source in enumerate(images(args.domain_validation / label)):
                methods[link(source, target_destination / f"domain_{index:05d}{source.suffix.lower()}")] += 1
                counts[f"target/domain/{label}"] += 1
            for source in images(raw_temporary / "val" / label):
                methods[link(source, target_destination / f"ckplus_{source.name}")] += 1
                counts[f"target/ckplus/{label}"] += 1

            test_destination = adaptation_temporary / "ckplus-test" / label
            test_destination.mkdir(parents=True)
            for source in images(raw_temporary / "test" / label):
                methods[link(source, test_destination / source.name)] += 1
                counts[f"ckplus/test/{label}"] += 1

        split_subject_ids = {
            split: sorted(subject for subject, value in assignment.items() if value == split)
            for split in ("train", "val", "test")
        }
        raw_counts = Counter(
            f"{split}/{label}"
            for path, split, label, _ in extracted
            if path not in rejected
        )
        report = {
            "source": {
                "repository": "AlirezaF138/ckplus-dataset",
                "revision": "059ad99127b13628dd23d1ce2f5f7a4259b1e8e8",
                "license_declared_by_repository": "cc0-1.0",
                "parquet": str(args.parquet),
                "sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
                "rows": len(rows),
                "usable_rows_before_deduplication": len(records),
                "skipped_labels": dict(skipped_labels),
            },
            "split": {
                "method": "subject-disjoint random search minimizing per-class ratio error",
                "seed": args.seed,
                "subjects": split_subject_ids,
                "subject_counts": {split: len(value) for split, value in split_subject_ids.items()},
                "image_counts": dict(sorted(raw_counts.items())),
            },
            "duplicate_audit": audit,
            "adaptation": {
                "base": str(args.base),
                "domain_validation": str(args.domain_validation),
                "repeat": args.repeat,
                "anger_fear_repeat": args.anger_fear_repeat,
                "counts": dict(sorted(counts.items())),
                "file_methods": dict(methods),
            },
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        (raw_temporary / "dataset.json").write_text(payload, encoding="utf-8")
        (adaptation_temporary / "dataset.json").write_text(payload, encoding="utf-8")
        raw_temporary.replace(args.raw_output)
        adaptation_temporary.replace(args.output)
        print(payload)
    except Exception:
        shutil.rmtree(raw_temporary, ignore_errors=True)
        shutil.rmtree(adaptation_temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
