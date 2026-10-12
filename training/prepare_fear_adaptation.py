from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path


CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def images(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fear-focused, identity-separated adaptation dataset")
    parser.add_argument("--base", type=Path, default=Path("data/seven-class/glasses-adaptation-v1"))
    parser.add_argument("--domain-validation", type=Path, default=Path("data/seven-class/ferplus-ravdess-cremad-kdef-v2/target_val"))
    parser.add_argument("--cremad", type=Path, default=Path("data/cremad/faces-facevote-v1"))
    parser.add_argument("--web", type=Path, default=Path("data/web-public/human-face-emotions/identity-split-v2/train"))
    parser.add_argument("--output", type=Path, default=Path("data/seven-class/fear-adaptation-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="fear-building-", dir=args.output.parent))
    counts: Counter[str] = Counter()
    try:
        for split in ("train", "val", "test"):
            for label in CLASSES:
                destination = temporary / split / label
                destination.mkdir(parents=True)
                for source in images(args.base / split / label):
                    link(source, destination / source.name)
                    counts[f"base/{split}/{label}"] += 1

        fear_destination = temporary / "train" / "fear"
        for source in images(args.cremad / "train" / "fear"):
            if source.stem.endswith(("_f0", "_f2")):
                link(source, fear_destination / f"cremad_fearfocus_{source.name}")
                counts["extra/cremad_fear"] += 1
        for repeat in range(4):
            for index, source in enumerate(images(args.web / "fear")):
                link(source, fear_destination / f"web_fearfocus_r{repeat}_{index:03d}{source.suffix.lower()}")
                counts["extra/web_fear"] += 1
        synthetic_fear = [
            source for source in images(args.base / "train" / "fear")
            if source.name.startswith("glasses_")
        ]
        for repeat in range(2):
            for index, source in enumerate(synthetic_fear):
                link(source, fear_destination / f"kdef_glassesfear_r{repeat}_{index:04d}.jpg")
                counts["extra/synthetic_glasses_fear"] += 1

        for label in CLASSES:
            destination = temporary / "target-val" / label
            destination.mkdir(parents=True)
            for index, source in enumerate(images(args.domain_validation / label)):
                link(source, destination / f"domain_{index:05d}{source.suffix.lower()}")
                counts[f"target/domain/{label}"] += 1
            for index, source in enumerate(images(args.base / "glasses-val" / label)):
                link(source, destination / f"glasses_{index:03d}{source.suffix.lower()}")
                counts[f"target/glasses/{label}"] += 1

        report = {
            "base": str(args.base),
            "training_additions": {
                "CREMA-D": "unused f0/f2 fear frames from training actors",
                "web": "manually curated, identity-split fear training images repeated four times",
                "synthetic_glasses": "existing fear training identities repeated twice",
            },
            "target_validation": "identity-separated RAVDESS, CREMA-D, KDEF, and synthetic-glasses validation images",
            "counts": dict(counts),
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
