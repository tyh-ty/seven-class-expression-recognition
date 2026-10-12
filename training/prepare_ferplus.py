from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from dataset_contract import CLASSES, SCHEMA_VERSION
FERPLUS_COLUMNS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
    "unknown",
    "NF",
)
FERPLUS_TO_CLASS = {
    "neutral": "neutral",
    "happiness": "happiness",
    "surprise": "surprise",
    "sadness": "sadness",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
}
USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}
SOURCES = {
    "train.parquet": {
        "url": "https://hf-mirror.com/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/train-00000-of-00001-5eab84e1c6a2fc27.parquet",
        "sha256": "b1c4bee933d41947d7d67648a54f3877de9365703f40117e0d59495c8eb6647a",
    },
    "publicTest.parquet": {
        "url": "https://hf-mirror.com/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
        "sha256": "e9efe94c45b5af29ec2825abb8f0bf3583450edb915b2c6ab6d5566afbffc5a1",
    },
    "privateTest.parquet": {
        "url": "https://hf-mirror.com/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
        "sha256": "e0b74317d8d3defc79961a297a651e23ee8b0785669484a258e7a389be1fcea0",
    },
    "fer2013new.csv": {
        "url": "https://cdn.jsdelivr.net/gh/microsoft/FERPlus@master/fer2013new.csv",
        "sha256": "9206e20d62f56475939f516847d61753e4860caeac9718b560129541b776fc2c",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FERPlus majority-vote train/val/test folders"
    )
    parser.add_argument("--raw", type=Path, default=Path("data/ferplus/raw-parquet"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/ferplus/majority-seven")
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading a missing raw file.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(source, headers={"User-Agent": "FERPlus trainer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_sources(raw: Path, no_download: bool) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, source in SOURCES.items():
        path = raw / name
        if not path.is_file():
            if no_download:
                raise FileNotFoundError(f"Missing raw dataset file: {path}")
            print(f"download {name}")
            download(source["url"], path)
        actual_hash = sha256(path)
        if actual_hash != source["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {source['sha256']}, got {actual_hash}"
            )
        hashes[name] = actual_hash
    return hashes


def load_parquet_images(path: Path) -> list[dict[str, Any]]:
    table = parquet.read_table(path, columns=["image", "label"])
    examples = table.to_pylist()
    for index, example in enumerate(examples):
        image = example.get("image") if isinstance(example, dict) else None
        if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
            raise TypeError(f"Invalid image example at {path}:{index}")
    return examples


def majority_label(row: dict[str, str]) -> str | None:
    votes = [int(row[column]) for column in FERPLUS_COLUMNS]
    votes = [0 if vote <= 1 else vote for vote in votes]
    total = sum(votes)
    winner_index = max(range(len(votes)), key=votes.__getitem__)
    if total == 0 or votes[winner_index] <= 0.5 * total or winner_index >= 8:
        return None
    return FERPLUS_TO_CLASS.get(FERPLUS_COLUMNS[winner_index])


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(
            f"Output is not empty: {args.output}. Use a new output path to avoid data loss."
        )
    hashes = ensure_sources(args.raw, args.no_download)
    examples_by_usage = {
        "Training": load_parquet_images(args.raw / "train.parquet"),
        "PublicTest": load_parquet_images(args.raw / "publicTest.parquet"),
        "PrivateTest": load_parquet_images(args.raw / "privateTest.parquet"),
    }
    with (args.raw / "fer2013new.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    image_count = sum(len(examples) for examples in examples_by_usage.values())
    if len(rows) != image_count:
        raise ValueError(
            f"Image/annotation count mismatch: {image_count} images, {len(rows)} rows"
        )
    for usage, examples in examples_by_usage.items():
        annotation_count = sum(row["Usage"] == usage for row in rows)
        if len(examples) != annotation_count:
            raise ValueError(
                f"{usage} count mismatch: {len(examples)} images, "
                f"{annotation_count} annotations"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"{args.output.name}-building-", dir=args.output.parent)
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    discarded: Counter[str] = Counter()
    usage_offsets: Counter[str] = Counter()
    try:
        for split in USAGE_TO_SPLIT.values():
            for label in CLASSES:
                (temporary_root / split / label).mkdir(parents=True, exist_ok=True)
        for row_index, row in enumerate(rows):
            usage = row["Usage"]
            if usage not in USAGE_TO_SPLIT:
                raise ValueError(f"Unknown FER2013 usage value: {usage}")
            example = examples_by_usage[usage][usage_offsets[usage]]
            usage_offsets[usage] += 1
            split = USAGE_TO_SPLIT[usage]
            label = majority_label(row)
            if label is None:
                discarded[split] += 1
                continue
            # FERPlus contains blank and repeated image-name fields. The row order is
            # the stable key shared with FER2013, so use it to prevent overwrites.
            destination = (
                temporary_root / split / label / f"ferplus_{row_index:07d}.jpg"
            )
            destination.write_bytes(example["image"]["bytes"])
            counts[split][label] += 1

        report = {
            "dataset": "FERPlus majority-vote",
            "classes": CLASSES,
            "split_mapping": USAGE_TO_SPLIT,
            "counts": {
                split: {
                    **{label: counts[split][label] for label in CLASSES},
                    "total": sum(counts[split].values()),
                    "discarded": discarded[split],
                }
                for split in ("train", "val", "test")
            },
            "source_hashes": hashes,
            "annotation_source": "Microsoft FERPlus fer2013new.csv",
            "image_source": "FER2013 train/publicTest/privateTest Parquet mirror",
            "alignment": "Parquet row order matched to FERPlus rows within each Usage split",
        }
        (temporary_root / "dataset.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.output.exists():
            args.output.rmdir()
        temporary_root.replace(args.output)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
