from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server
from benchmark_glasses_product import MAPPING, add_glasses, detector, primary_face


def save_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not success:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(path)


def crop_product_face(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    x, y, width, height = [int(round(value)) for value in face[:4]]
    left, top, crop_width, crop_height = server.EmotionRecognizer._square_face_box(
        x, y, width, height, image.shape[1], image.shape[0]
    )
    crop = image[top : top + crop_height, left : left + crop_width]
    return cv2.resize(
        crop,
        (224, 224),
        interpolation=cv2.INTER_AREA if min(crop.shape[:2]) >= 224 else cv2.INTER_LINEAR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make identity-separated synthetic glasses adaptation images")
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "seven-class" / "ferplus-ravdess-cremad-kdef-v2")
    parser.add_argument("--archive", type=Path, default=Path(r"D:\KDEF\KDEF_and_AKDEF.zip"))
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "seven-class" / "glasses-adaptation-v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="glasses-building-", dir=args.output.parent))
    counts: Counter[str] = Counter()
    pattern = re.compile(r"[AB][FM](\d{2})(AN|DI|AF|HA|NE|SA|SU)S")
    face_detector = detector(0.35)
    try:
        for split in ("train", "val", "test"):
            for label, _ in server.EMOTIONS:
                destination = temporary / split / label
                destination.mkdir(parents=True)
                for path in (args.base / split / label).iterdir():
                    if path.is_file():
                        os.link(path, destination / path.name)
                        counts[f"base/{split}/{label}"] += 1

        with ZipFile(args.archive) as archive:
            for entry in sorted(archive.namelist()):
                match = pattern.fullmatch(Path(entry).stem.upper())
                if not match:
                    continue
                person_number = int(match.group(1))
                if person_number <= 25:
                    split = "train"
                elif person_number <= 30:
                    split = "glasses-val"
                else:
                    continue
                label = MAPPING[match.group(2)]
                image = cv2.imdecode(
                    np.frombuffer(archive.read(entry), dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise RuntimeError(f"Cannot decode {entry}")
                scale = min(1.0, 512 / max(image.shape[:2]))
                if scale < 1.0:
                    image = cv2.resize(
                        image,
                        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                face = primary_face(face_detector, image)
                if face is None:
                    counts[f"missed/{split}/{label}"] += 1
                    continue
                landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
                variants = 3 if split == "train" else 1
                for variant in range(variants):
                    seed = person_number * 7 + variant * 41
                    altered = add_glasses(image, landmarks, seed)
                    crop = crop_product_face(altered, face)
                    destination = temporary / split / label
                    destination.mkdir(parents=True, exist_ok=True)
                    output_name = f"glasses_{Path(entry).stem.lower()}_v{variant}.jpg"
                    save_image(destination / output_name, crop)
                    counts[f"synthetic/{split}/{label}"] += 1
        if sum(counts[f"synthetic/glasses-val/{label}"] for label, _ in server.EMOTIONS) != 140:
            raise RuntimeError("Glasses validation should have 140 examples")
        report = {
            "base": str(args.base),
            "archive": str(args.archive),
            "split": "KDEF people 01–25 for training, 26–30 for validation, 31–35 untouched final test",
            "synthetic_note": "Landmark-placed oval glasses and glare are an approximation, not genuine eyewear photos",
            "counts": dict(counts),
        }
        (temporary / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
