from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import EmotionRecognizer, FACE_DETECTOR_PATH


CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
FACE_VOTE_LABELS = {
    "A": "anger",
    "D": "disgust",
    "F": "fear",
    "H": "happiness",
    "N": "neutral",
    "S": "sadness",
}
INTENDED_LABELS = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness",
}
SPLITS = {
    "train": set(range(1001, 1072)),
    "val": set(range(1072, 1082)),
    "test": set(range(1082, 1092)),
}
FRAME_POSITIONS = (0.40, 0.55, 0.70)
KNOWN_BROKEN = {
    "1064_IEO_DIS_MD",
    "1064_TIE_SAD_XX",
    "1076_MTI_NEU_XX",
    "1076_MTI_SAD_XX",
}
_WORKER_DETECTOR: cv2.FaceDetectorYN | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract identity-separated face crops from CREMA-D videos"
    )
    parser.add_argument(
        "--input", type=Path, default=Path(r"D:\CREMA-D-gitlab\VideoFlash")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path(r"D:\CREMA-D-gitlab\summaryTable.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/cremad/faces-facevote-v1")
    )
    parser.add_argument("--detector", type=Path, default=FACE_DETECTOR_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--frames-per-video", type=int, default=3, choices=range(1, len(FRAME_POSITIONS) + 1)
    )
    parser.add_argument(
        "--minimum-face-confidence",
        type=float,
        default=50.0,
        help="Minimum percentage attached to an unambiguous FaceVote label.",
    )
    parser.add_argument(
        "--label-source",
        choices=("face-vote", "intended"),
        default="face-vote",
        help="Use the visual crowd vote or the emotion code in the filename.",
    )
    return parser.parse_args()


def split_for(actor: int) -> str:
    for name, actors in SPLITS.items():
        if actor in actors:
            return name
    raise ValueError(f"Actor {actor} is outside the CREMA-D identity split")


def load_face_votes(path: Path) -> dict[str, tuple[str | None, float | None]]:
    votes: dict[str, tuple[str | None, float | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            stem = row["FileName"].strip()
            raw_vote = row["FaceVote"].strip()
            label = FACE_VOTE_LABELS.get(raw_vote)
            try:
                confidence = float(row["FaceLevel"])
            except (TypeError, ValueError):
                confidence = None
            votes[stem] = (label, confidence)
    return votes


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
    video_path: Path,
    output_root: Path,
    detector: cv2.FaceDetectorYN,
    label: str,
    actor: int,
    frames_per_video: int,
) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or frame_count <= 0:
        capture.release()
        return 0, frames_per_video
    saved = 0
    attempted = 0
    try:
        for frame_index, position in enumerate(FRAME_POSITIONS[:frames_per_video]):
            attempted += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((frame_count - 1) * position)))
            success, frame = capture.read()
            if not success or frame is None:
                continue
            crop = detect_face(detector, frame)
            if crop is None:
                continue
            split = split_for(actor)
            destination = (
                output_root
                / split
                / label
                / f"cremad_actor{actor}_{video_path.stem}_f{frame_index}.jpg"
            )
            encoded, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not encoded:
                continue
            jpeg.tofile(str(destination))
            saved += 1
    finally:
        capture.release()
    return saved, attempted


def initialize_worker(detector_path: str) -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = cv2.FaceDetectorYN.create(
        detector_path, "", (320, 320), 0.65, 0.3, 5000
    )


def process_task(task: tuple[str, str, str, int, int]) -> tuple[str, str, int, int]:
    video_path, output_root, label, actor, frames_per_video = task
    if _WORKER_DETECTOR is None:
        raise RuntimeError("Face detector worker was not initialized")
    saved, attempted = process_video(
        Path(video_path),
        Path(output_root),
        _WORKER_DETECTOR,
        label,
        actor,
        frames_per_video,
    )
    return video_path, label, saved, attempted


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.minimum_face_confidence < 0 or args.minimum_face_confidence > 100:
        raise ValueError("--minimum-face-confidence must be between 0 and 100")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    votes = load_face_votes(args.summary)
    videos = sorted(args.input.glob("*.flv"))
    if len(videos) < 7400:
        raise RuntimeError(
            f"CREMA-D download is incomplete: found {len(videos)} videos, expected at least 7400"
        )

    detector_path = EmotionRecognizer._opencv_safe_path(args.detector.resolve())
    for split in SPLITS:
        for label in CLASSES:
            (args.output / split / label).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    failures: list[str] = []
    attempted_frames = 0
    tasks: list[tuple[str, str, str, int, int]] = []
    for video_path in videos:
        fields = video_path.stem.split("_")
        if len(fields) != 4 or fields[2] not in INTENDED_LABELS:
            skipped["unrecognized_filename"] += 1
            continue
        if video_path.stem in KNOWN_BROKEN:
            skipped["known_broken_video"] += 1
            continue
        actor = int(fields[0])
        if args.label_source == "face-vote":
            label, confidence = votes.get(video_path.stem, (None, None))
            if label is None:
                skipped["ambiguous_or_missing_face_vote"] += 1
                continue
            if confidence is None or confidence < args.minimum_face_confidence:
                skipped["low_face_vote_confidence"] += 1
                continue
        else:
            label = INTENDED_LABELS[fields[2]]
        tasks.append(
            (str(video_path), str(args.output), label, actor, args.frames_per_video)
        )

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(detector_path),),
    ) as executor:
        results = executor.map(process_task, tasks, chunksize=8)
        for index, (video_path_string, label, saved, attempted) in enumerate(results, start=1):
            video_path = Path(video_path_string)
            actor = int(video_path.stem.split("_", 1)[0])
            attempted_frames += attempted
            split = split_for(actor)
            counts[f"{split}/{label}"] += saved
            counts[f"{split}/videos"] += int(saved > 0)
            if saved == 0:
                failures.append(video_path.name)
            if index % 250 == 0 or index == len(tasks):
                print(
                    f"videos={index}/{len(tasks)} saved_frames={sum(v for k, v in counts.items() if '/' in k and not k.endswith('/videos'))}",
                    flush=True,
                )

    report = {
        "source": str(args.input.resolve()),
        "summary": str(args.summary.resolve()),
        "classes": CLASSES,
        "label_source": args.label_source,
        "minimum_face_confidence": args.minimum_face_confidence,
        "actor_splits": {name: sorted(actors) for name, actors in SPLITS.items()},
        "frame_positions": FRAME_POSITIONS[: args.frames_per_video],
        "known_broken_excluded": sorted(KNOWN_BROKEN),
        "source_videos": len(videos),
        "attempted_frames": attempted_frames,
        "saved_frames": sum(
            value for key, value in counts.items() if not key.endswith("/videos")
        ),
        "counts": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "videos_without_a_detected_face": failures,
    }
    (args.output / "dataset.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for split in SPLITS:
        for label in CLASSES[:-1]:
            if not list((args.output / split / label).glob("*.jpg")):
                raise RuntimeError(f"No extracted face frames in {split}/{label}")


if __name__ == "__main__":
    main()
