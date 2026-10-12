from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import EmotionRecognizer, FACE_DETECTOR_PATH


CLASSES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect square face crops locally")
    parser.add_argument("--subject", required=True, help="Anonymous subject id, for example p001")
    parser.add_argument("--label", choices=CLASSES, default="neutral")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/custom/raw"))
    parser.add_argument("--auto-interval", type=float, default=0.6)
    parser.add_argument("--target-per-label", type=int, default=80)
    parser.add_argument("--min-blur-score", type=float, default=40.0)
    parser.add_argument("--min-frame-difference", type=float, default=1.5)
    return parser.parse_args()


def safe_subject(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", value.strip())
    if not cleaned:
        raise ValueError("Subject id is empty after sanitization")
    return cleaned


def square_face_crop(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image_height, image_width = image.shape[:2]
    left, top, crop_width, crop_height = EmotionRecognizer._square_face_box(
        x,
        y,
        width,
        height,
        image_width,
        image_height,
    )
    crop = image[top : top + crop_height, left : left + crop_width]
    interpolation = (
        cv2.INTER_AREA
        if crop_width >= 224 and crop_height >= 224
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(crop, (224, 224), interpolation=interpolation)
    return resized, (left, top, crop_width, crop_height)


def append_manifest(path: Path, row: list[str]) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(
                [
                    "file",
                    "subject",
                    "label",
                    "captured_at",
                    "sharpness",
                    "crop_method",
                ]
            )
        writer.writerow(row)


def save_jpeg(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("Unable to encode JPEG")
    encoded.tofile(str(path))


def opencv_safe_model_path(path: Path) -> Path:
    try:
        str(path).encode("ascii")
        return path
    except UnicodeEncodeError:
        cache = Path(tempfile.gettempdir()) / "emotion-recognition-demo"
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / path.name
        if not destination.is_file() or destination.stat().st_size != path.stat().st_size:
            shutil.copy2(path, destination)
        return destination


def main() -> None:
    args = parse_args()
    if args.auto_interval <= 0:
        raise ValueError("--auto-interval must be positive")
    if args.target_per_label < 1:
        raise ValueError("--target-per-label must be positive")
    if args.min_frame_difference < 0:
        raise ValueError("--min-frame-difference cannot be negative")
    subject = safe_subject(args.subject)
    current_label = args.label
    for label in CLASSES:
        (args.output / subject / label).mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest_square_v1.csv"
    detector_path = opencv_safe_model_path(FACE_DETECTOR_PATH)
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), 0.65, 0.3, 5000
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera}")
    saved_counts = {
        label: len(list((args.output / subject / label).glob("*.jpg")))
        for label in CLASSES
    }
    auto_capture = False
    last_auto_capture = 0.0
    last_saved_gray: dict[str, np.ndarray] = {}

    def save_face(crop: np.ndarray, label: str, sharpness: float) -> None:
        captured_at = datetime.now().astimezone()
        filename = captured_at.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        output_path = args.output / subject / label / filename
        save_jpeg(output_path, crop)
        append_manifest(
            manifest,
            [
                str(output_path),
                subject,
                label,
                captured_at.isoformat(),
                f"{sharpness:.3f}",
                "yunet-square-1.45",
            ],
        )
        saved_counts[label] += 1
        last_saved_gray[label] = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)
            face_crop = None
            blur_score = 0.0
            frame_difference = float("inf")
            if faces is not None and len(faces):
                face = max(faces, key=lambda item: item[2] * item[3])
                x, y, box_width, box_height = [int(round(value)) for value in face[:4]]
                face_crop, display_box = square_face_crop(
                    frame,
                    x,
                    y,
                    box_width,
                    box_height,
                )
                grayscale = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
                previous_gray = last_saved_gray.get(current_label)
                if previous_gray is not None:
                    frame_difference = float(
                        np.mean(cv2.absdiff(grayscale, previous_gray))
                    )
                left, top, crop_width, crop_height = display_box
                cv2.rectangle(
                    frame,
                    (left, top),
                    (left + crop_width, top + crop_height),
                    (15, 159, 112),
                    2,
                )
            cv2.putText(
                frame,
                (
                    f"subject={subject} label={current_label} "
                    f"saved={saved_counts[current_label]}/{args.target_per_label}"
                ),
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"A: auto {'ON' if auto_capture else 'OFF'}  SPACE: save  Q: quit",
                (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "1 anger  2 disgust  3 fear  4 happiness",
                (16, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "5 neutral  6 sadness  7 surprise",
                (16, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            quality_color = (15, 159, 112) if blur_score >= args.min_blur_score else (0, 165, 255)
            cv2.putText(
                frame,
                f"sharpness={blur_score:.0f}  change={frame_difference:.1f}",
                (16, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                quality_color,
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Emotion dataset collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("7"):
                current_label = CLASSES[key - ord("1")]
                auto_capture = False
            elif key == ord("a"):
                auto_capture = not auto_capture
                last_auto_capture = 0.0

            sharp_enough = face_crop is not None and blur_score >= args.min_blur_score
            manual_save = key == 32 and sharp_enough
            now = perf_counter()
            automatic_save = (
                auto_capture
                and sharp_enough
                and frame_difference >= args.min_frame_difference
                and saved_counts[current_label] < args.target_per_label
                and now - last_auto_capture >= args.auto_interval
            )
            if manual_save or automatic_save:
                assert face_crop is not None
                save_face(face_crop, current_label, blur_score)
                last_auto_capture = now
                if saved_counts[current_label] >= args.target_per_label:
                    auto_capture = False
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
