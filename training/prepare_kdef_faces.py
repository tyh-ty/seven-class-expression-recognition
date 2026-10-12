from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import EmotionRecognizer, FACE_DETECTOR_PATH


CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
EMOTION_CODES = {
    "AN": "anger",
    "DI": "disgust",
    "AF": "fear",
    "HA": "happiness",
    "NE": "neutral",
    "SA": "sadness",
    "SU": "surprise",
}
FILENAME_PATTERN = re.compile(
    r"(?P<session>[AB])(?P<gender>[FM])(?P<person>\d{2})"
    r"(?P<emotion>AN|DI|AF|HA|NE|SA|SU)(?P<angle>FL|HL|HR|FR|S)"
)
SPLITS = {
    "train": set(range(1, 26)),
    "val": set(range(26, 31)),
    "test": set(range(31, 36)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare identity-separated KDEF face crops for seven-class training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"D:\KDEF\KDEF_and_AKDEF.zip"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/kdef/faces-seven-v2")
    )
    parser.add_argument("--detector", type=Path, default=FACE_DETECTOR_PATH)
    parser.add_argument(
        "--angles",
        nargs="+",
        choices=("S", "HL", "HR", "FL", "FR"),
        default=("S", "HL", "HR"),
        help="Camera angles to keep. S is straight; H is half profile; F is full profile.",
    )
    return parser.parse_args()


def split_for(person_number: int) -> str:
    for split, numbers in SPLITS.items():
        if person_number in numbers:
            return split
    raise ValueError(f"KDEF person number is outside the expected range: {person_number}")


def crop_face(detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None:
        return None
    valid = [face for face in faces if face[2] >= 40 and face[3] >= 40]
    if not valid:
        return None
    face = max(valid, key=lambda item: float(item[2] * item[3]))
    x, y, box_width, box_height = [int(round(value)) for value in face[:4]]
    left, top, crop_width, crop_height = EmotionRecognizer._square_face_box(
        x, y, box_width, box_height, width, height
    )
    crop = image[top : top + crop_height, left : left + crop_width]
    if crop.size == 0:
        return None
    interpolation = cv2.INTER_AREA if min(crop.shape[:2]) >= 224 else cv2.INTER_LINEAR
    return cv2.resize(crop, (224, 224), interpolation=interpolation)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    detector_path = EmotionRecognizer._opencv_safe_path(args.detector.resolve())
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), 0.5, 0.3, 5000
    )
    for split in SPLITS:
        for label in CLASSES:
            (args.output / split / label).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    identities: dict[str, set[str]] = {split: set() for split in SPLITS}
    failures: list[str] = []
    ignored: Counter[str] = Counter()
    selected_angles = set(args.angles)
    with ZipFile(args.input) as archive:
        entries = sorted(
            (entry for entry in archive.infolist() if not entry.is_dir()),
            key=lambda entry: entry.filename,
        )
        for entry in entries:
            suffix = Path(entry.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                ignored["non_image"] += 1
                continue
            stem = Path(entry.filename).stem.upper()
            match = FILENAME_PATTERN.fullmatch(stem)
            if match is None:
                ignored["non_kdef_image"] += 1
                continue
            fields = match.groupdict()
            angle = fields["angle"]
            if angle not in selected_angles:
                ignored[f"angle_{angle}"] += 1
                continue
            person_number = int(fields["person"])
            split = split_for(person_number)
            label = EMOTION_CODES[fields["emotion"]]
            identity = f"{fields['gender']}{person_number:02d}"
            identities[split].add(identity)

            encoded = np.frombuffer(archive.read(entry), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                failures.append(entry.filename)
                continue
            crop = crop_face(detector, image)
            if crop is None:
                failures.append(entry.filename)
                continue
            destination = args.output / split / label / f"kdef_{stem.lower()}.jpg"
            success, jpeg = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not success:
                failures.append(entry.filename)
                continue
            jpeg.tofile(str(destination))
            counts[f"{split}/{label}"] += 1

    report = {
        "source": str(args.input.resolve()),
        "classes": CLASSES,
        "emotion_mapping": EMOTION_CODES,
        "angles": list(args.angles),
        "identity_definition": "gender plus person number; session A/B kept in the same split",
        "identity_splits": {
            split: sorted(values) for split, values in identities.items()
        },
        "person_number_splits": {
            split: sorted(values) for split, values in SPLITS.items()
        },
        "counts": dict(sorted(counts.items())),
        "saved_images": sum(counts.values()),
        "failed_images": failures,
        "ignored": dict(sorted(ignored.items())),
    }
    (args.output / "dataset.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    for split in SPLITS:
        for label in CLASSES:
            if counts[f"{split}/{label}"] == 0:
                raise RuntimeError(f"No KDEF images were prepared for {split}/{label}")
    if failures:
        print(f"warning: {len(failures)} KDEF images could not be decoded or cropped")


if __name__ == "__main__":
    main()
