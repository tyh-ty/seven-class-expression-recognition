from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


@dataclass
class Sample:
    path: Path
    relative_path: str
    label: str
    feature: np.ndarray | None
    detection_score: float
    group: int = -1
    split: str = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an identity-grouped split for curated web face images"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--identity-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def extract_feature(
    image: np.ndarray,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> tuple[np.ndarray | None, float]:
    height, width = image.shape[:2]
    scale = max(1.0, 160.0 / min(width, height))
    if scale > 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None, 0.0
    face = max(faces, key=lambda row: float(row[-1]))
    aligned = recognizer.alignCrop(image, face)
    feature = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm <= 0:
        return None, float(face[-1])
    return feature / norm, float(face[-1])


def discover_samples(
    root: Path,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> list[Sample]:
    samples: list[Sample] = []
    for label in CLASSES:
        class_root = root / label
        if not class_root.is_dir():
            continue
        for path in sorted(class_root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            feature, score = extract_feature(read_image(path), detector, recognizer)
            samples.append(
                Sample(
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    label=label,
                    feature=feature,
                    detection_score=score,
                )
            )
    return samples


def assign_identity_groups(samples: list[Sample], threshold: float) -> None:
    feature_groups: list[list[int]] = []
    for index, sample in enumerate(samples):
        if sample.feature is None:
            sample.group = len(feature_groups)
            feature_groups.append([index])
            continue
        candidates: list[tuple[float, int]] = []
        for group_index, members in enumerate(feature_groups):
            member_features = [
                samples[member].feature
                for member in members
                if samples[member].feature is not None
            ]
            if not member_features:
                continue
            similarities = [
                float(np.dot(sample.feature, feature))
                for feature in member_features
                if feature is not None
            ]
            if similarities and min(similarities) >= threshold:
                candidates.append((sum(similarities) / len(similarities), group_index))
        if candidates:
            _, selected = max(candidates)
            sample.group = selected
            feature_groups[selected].append(index)
        else:
            sample.group = len(feature_groups)
            feature_groups.append([index])


def choose_validation_groups(
    samples: list[Sample], validation_fraction: float, seed: int
) -> dict[str, int]:
    totals = Counter(sample.label for sample in samples)
    targets = {
        label: (
            min(totals[label] - 1, max(1, round(totals[label] * validation_fraction)))
            if totals[label] >= 2
            else 0
        )
        for label in CLASSES
    }
    groups: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)
    group_counts = {
        group: Counter(sample.label for sample in group_samples)
        for group, group_samples in groups.items()
    }
    randomizer = random.Random(seed)
    tie_breakers = {group: randomizer.random() for group in groups}
    validation_counts: Counter[str] = Counter()
    remaining = set(groups)

    while remaining:
        before_error = sum(
            abs(targets[label] - validation_counts[label]) for label in CLASSES
        )
        candidates: list[tuple[float, float, int]] = []
        for group in remaining:
            counts = group_counts[group]
            if any(
                totals[label] >= 2
                and totals[label] - validation_counts[label] - counts[label] < 1
                for label in CLASSES
            ):
                continue
            after_error = sum(
                abs(targets[label] - validation_counts[label] - counts[label])
                for label in CLASSES
            )
            improvement = before_error - after_error
            if improvement > 0:
                candidates.append((improvement, tie_breakers[group], group))
        if not candidates:
            break
        _, _, selected = max(candidates)
        remaining.remove(selected)
        validation_counts.update(group_counts[selected])
        for sample in groups[selected]:
            sample.split = "val"
    return targets


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_split(samples: list[Sample], output: Path) -> Counter[str]:
    methods: Counter[str] = Counter()
    for split in ("train", "val"):
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        destination = (
            output
            / sample.split
            / sample.label
            / f"web_{index:04d}_{sample.path.name}"
        )
        methods[link_or_copy(sample.path, destination)] += 1
    return methods


def write_manifest(samples: list[Sample], output: Path) -> None:
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "source_path",
                "label",
                "identity_group",
                "split",
                "face_feature_available",
                "detection_score",
            ),
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "source_path": sample.relative_path,
                    "label": sample.label,
                    "identity_group": sample.group,
                    "split": sample.split,
                    "face_feature_available": sample.feature is not None,
                    "detection_score": f"{sample.detection_score:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    for path in (args.input, args.detector, args.recognizer):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new output path to avoid data loss."
        )

    detector = cv2.FaceDetectorYN.create(
        str(args.detector),
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(args.recognizer), "")
    samples = discover_samples(args.input, detector, recognizer)
    if not samples:
        raise ValueError(f"No images found under {args.input}")
    assign_identity_groups(samples, args.identity_threshold)
    targets = choose_validation_groups(
        samples, args.validation_fraction, args.seed
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{args.output.name}-building-", dir=args.output.parent)
    )
    try:
        methods = write_split(samples, temporary)
        write_manifest(samples, temporary)
        groups: dict[int, list[Sample]] = defaultdict(list)
        for sample in samples:
            groups[sample.group].append(sample)
        report = {
            "dataset": "Manually curated public web face images",
            "source": str(args.input),
            "total": len(samples),
            "face_features_available": sum(
                sample.feature is not None for sample in samples
            ),
            "identity_groups": len(groups),
            "multi_image_identity_groups": sum(
                len(group) > 1 for group in groups.values()
            ),
            "largest_identity_group": max(len(group) for group in groups.values()),
            "identity_threshold": args.identity_threshold,
            "validation_fraction": args.validation_fraction,
            "validation_targets": targets,
            "counts": {
                split: {
                    label: sum(
                        sample.split == split and sample.label == label
                        for sample in samples
                    )
                    for label in CLASSES
                }
                for split in ("train", "val")
            },
            "privacy": "Face features were used in memory for grouping and were not saved.",
            "limitations": "Identity grouping is automatic and may contain false matches or misses.",
            "file_methods": dict(methods),
        }
        (temporary / "report.json").write_text(
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
