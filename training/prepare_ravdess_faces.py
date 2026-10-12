from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import EmotionRecognizer, FACE_DETECTOR_PATH


CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
EMOTION_CODES = {
    "01": "neutral",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
SPLITS = {
    "train": set(range(1, 17)),
    "val": set(range(17, 21)),
    "test": set(range(21, 25)),
}
FRAME_POSITIONS = (0.35, 0.45, 0.55, 0.65, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract identity-separated face frames from RAVDESS speech videos")
    parser.add_argument("--input", type=Path, default=Path(r"D:\RAVDESS"))
    parser.add_argument("--output", type=Path, default=Path("data/ravdess/faces-seven"))
    parser.add_argument("--detector", type=Path, default=FACE_DETECTOR_PATH)
    parser.add_argument("--frames-per-video", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--actors", type=int, nargs="+", default=list(range(1, 25)))
    return parser.parse_args()


def split_for(actor: int) -> str:
    return next(name for name, actors in SPLITS.items() if actor in actors)


def detect_face(detector: cv2.FaceDetectorYN, frame: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
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
    crop = frame[top : top + crop_height, left : left + crop_width]
    if crop.size == 0:
        return None
    interpolation = cv2.INTER_AREA if min(crop.shape[:2]) >= 224 else cv2.INTER_LINEAR
    return cv2.resize(crop, (224, 224), interpolation=interpolation)


def process_video(
    archive: ZipFile,
    entry_name: str,
    output_root: Path,
    detector: cv2.FaceDetectorYN,
    frames_per_video: int,
    temporary_root: Path,
) -> tuple[int, int]:
    filename = Path(entry_name).name
    fields = filename.removesuffix(".mp4").split("-")
    if len(fields) != 7 or fields[0] != "01" or fields[1] != "01":
        return 0, 0
    label = EMOTION_CODES.get(fields[2])
    if label is None:
        return 0, 0
    actor = int(fields[6])
    split = split_for(actor)
    video_path = temporary_root / filename
    with archive.open(entry_name) as source, video_path.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = FRAME_POSITIONS[:frames_per_video]
    saved = 0
    attempted = 0
    try:
        for frame_index, position in enumerate(positions):
            attempted += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((frame_count - 1) * position)))
            success, frame = capture.read()
            if not success or frame is None:
                continue
            crop = detect_face(detector, frame)
            if crop is None:
                continue
            destination = output_root / split / label / f"actor{actor:02d}_{filename[:-4]}_f{frame_index}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not encoded:
                continue
            jpeg.tofile(str(destination))
            saved += 1
    finally:
        capture.release()
        video_path.unlink(missing_ok=True)
    return saved, attempted


def main() -> None:
    args = parse_args()
    if any(actor not in range(1, 25) for actor in args.actors):
        raise ValueError("--actors must be between 1 and 24")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    detector_path = EmotionRecognizer._opencv_safe_path(args.detector.resolve())
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.65, 0.3, 5000)
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    processed_videos = 0
    attempted_frames = 0
    args.output.mkdir(parents=True)
    for split in SPLITS:
        for label in CLASSES:
            (args.output / split / label).mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="ravdess-face-") as temporary:
        temporary_root = Path(temporary)
        for actor in sorted(set(args.actors)):
            archive_path = args.input / f"Video_Speech_Actor_{actor:02d}.zip"
            recovered_archive = args.input / f"Video_Speech_Actor_{actor:02d}.av-seven-recovered.zip"
            if recovered_archive.is_file():
                archive_path = recovered_archive
            try:
                with ZipFile(archive_path) as archive:
                    entries = sorted(name for name in archive.namelist() if name.lower().endswith(".mp4"))
                    for entry_name in entries:
                        fields = Path(entry_name).stem.split("-")
                        if len(fields) != 7 or fields[0] != "01" or fields[2] not in EMOTION_CODES:
                            continue
                        saved, attempted = process_video(
                            archive, entry_name, args.output, detector, args.frames_per_video, temporary_root
                        )
                        processed_videos += 1
                        attempted_frames += attempted
                        counts[f"{split_for(actor)}/{EMOTION_CODES[fields[2]]}"] += saved
                print(f"actor {actor:02d}: complete", flush=True)
            except (FileNotFoundError, BadZipFile, OSError) as error:
                failures.append({"actor": f"{actor:02d}", "archive": str(archive_path), "error": str(error)})
                print(f"actor {actor:02d}: skipped ({error})", flush=True)

    report = {
        "source": str(args.input.resolve()),
        "selected_actors": sorted(set(args.actors)),
        "recovered_archives": [
            str(args.input / f"Video_Speech_Actor_{actor:02d}.av-seven-recovered.zip")
            for actor in sorted(set(args.actors))
            if (args.input / f"Video_Speech_Actor_{actor:02d}.av-seven-recovered.zip").is_file()
        ],
        "classes": CLASSES,
        "emotion_mapping": EMOTION_CODES,
        "actor_splits": {name: sorted(actors) for name, actors in SPLITS.items()},
        "video_modality": "01 full audiovisual speech (visual frames only)",
        "excluded": ["02 calm", "duplicate 02 video-only modality", "song videos"],
        "frame_positions": FRAME_POSITIONS[: args.frames_per_video],
        "processed_videos": processed_videos,
        "attempted_frames": attempted_frames,
        "saved_frames": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "failures": failures,
    }
    (args.output / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    selected_splits = {split_for(actor) for actor in args.actors}
    if any(not list((args.output / split / label).glob("*.jpg")) for split in selected_splits for label in CLASSES):
        raise RuntimeError("At least one split/class has no extracted face frames")


if __name__ == "__main__":
    main()
