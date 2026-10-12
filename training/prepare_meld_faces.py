"""Extract conservative single-face MELD examples from the authors' video archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
LABELS = {"anger": "anger", "disgust": "disgust", "fear": "fear", "joy": "happiness",
          "neutral": "neutral", "sadness": "sadness", "surprise": "surprise"}
SPLITS = {"train": "train_splits", "val": "dev_splits_complete", "test": "output_repeated_splits_test"}
THREAD = threading.local()
VIDEO_NAME = re.compile(r"(?:\./)?(?:train_splits|dev_splits_complete|dev_splits|output_repeated_splits_test|test_splits)/dia(\d+)_utt(\d+)\.mp4$")


def rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return {
            f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4": row
            for row in csv.DictReader(source)
        }


def select(rows_by_name: dict[str, dict], split: str) -> set[str]:
    selected = set()
    for emotion in LABELS:
        candidates = [name for name, row in rows_by_name.items()
                      if row["Emotion"] == emotion and row["Speaker"] != "All"]
        candidates.sort(key=lambda name: hashlib.sha256(
            ("MELD-seed42:" + split + ":" + name).encode("ascii")
        ).digest())
        cap = (450 if emotion == "neutral" else 400) if split == "train" else (
            200 if emotion == "neutral" else 120
        ) if split == "test" else None
        selected.update(candidates[:cap])
    return selected


def detector() -> cv2.FaceDetectorYN:
    if not hasattr(THREAD, "detector"):
        model_path = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
        try:
            str(model_path).encode("ascii")
        except UnicodeEncodeError:
            cache = Path(tempfile.gettempdir()) / "emotion-recognition-training"
            cache.mkdir(parents=True, exist_ok=True)
            cached = cache / model_path.name
            if not cached.is_file() or cached.stat().st_size != model_path.stat().st_size:
                shutil.copy2(model_path, cached)
            model_path = cached
        THREAD.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "", (320, 320), 0.72, 0.3, 5000
        )
    return THREAD.detector


def crop_face(frame: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    scale = min(1.0, 720.0 / max(height, width))
    if scale < 1:
        frame = cv2.resize(frame, (round(width*scale), round(height*scale)))
    height, width = frame.shape[:2]
    face_detector = detector()
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(frame)
    if faces is None or len(faces) != 1:
        return None
    face = faces[0]
    x, y, w, h = [int(round(v)) for v in face[:4]]
    if min(w, h) < 55 or float(face[-1]) < 0.75:
        return None
    size = round(max(w, h) * 1.25)
    cx = x+w/2
    cy = y+h/2
    left = max(0, min(width-size, round(cx-size/2)))
    top = max(0, min(height-size, round(cy-size/2)))
    if size > min(height, width):
        return None
    result = frame[top:top+size, left:left+size]
    if result.size == 0:
        return None
    return cv2.resize(result, (224, 224), interpolation=cv2.INTER_AREA)


def process_video(raw: bytes, output: Path, split: str, label: str, stem: str) -> tuple[str, int]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=r"D:\MELD-source", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(raw)
    saved = 0
    try:
        capture = cv2.VideoCapture(str(temporary))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count <= 0:
            return "bad_video", 0
        positions = (0.40, 0.65) if split == "train" else (0.55,)
        for frame_number, position in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(count-1, round((count-1)*position)))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            crop = crop_face(frame)
            if crop is None:
                continue
            encoded, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
            if not encoded:
                continue
            jpeg.tofile(str(output / split / label / f"meld_{stem}_f{frame_number}.jpg"))
            saved += 1
        capture.release()
        return "saved" if saved else "face_not_unique", saved
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(r"D:\MELD-source\MELD.Raw.tar.gz"))
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "meld" / "annotations")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "meld" / "faces-v1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--per-class-cap",
        type=int,
        help="Override the normal per-class cap in every split (useful for a QA sample).",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary_output = args.output.with_name("." + args.output.name + "-building")
    if temporary_output.exists():
        raise FileExistsError(temporary_output)
    labels = {s: rows(args.annotations / f"{('dev' if s == 'val' else s)}_sent_emo.csv")
              for s in ("train", "val", "test")}
    selected = {s: select(labels[s], s) for s in labels}
    if args.per_class_cap is not None:
        if args.per_class_cap < 1:
            raise ValueError("--per-class-cap must be positive")
        for split in selected:
            selected[split] = {
                name
                for emotion in LABELS
                for name in sorted(
                    (
                        candidate
                        for candidate in selected[split]
                        if labels[split][candidate]["Emotion"] == emotion
                    ),
                    key=lambda candidate: hashlib.sha256(
                        ("MELD-QA:" + split + ":" + candidate).encode("ascii")
                    ).digest(),
                )[: args.per_class_cap]
            }
    for split in labels:
        for label in CLASSES:
            (temporary_output / split / label).mkdir(parents=True, exist_ok=True)
    print("requested", {k:len(v) for k,v in selected.items()}, flush=True)
    counts: Counter[str] = Counter()
    pending = set()

    def collect(done):
        for future in done:
            split, label, speaker, result, n = future.result()
            counts[f"{split}/{label}/{result}/videos"] += 1
            counts[f"{split}/{label}/frames"] += n
            counts[f"{split}/speaker/{speaker}/frames"] += n

    with ThreadPoolExecutor(max_workers=args.workers) as pool, tarfile.open(args.archive, "r|gz") as outer:
        for outer_member in outer:
            split = next((s for s in labels if outer_member.name == f"MELD.Raw/{'dev' if s == 'val' else s}.tar.gz"), None)
            if split is None:
                continue
            print("archive_split", split, outer_member.name, flush=True)
            source = outer.extractfile(outer_member)
            if source is None:
                raise RuntimeError(f"Cannot read {outer_member.name}")
            with source, tarfile.open(fileobj=source, mode="r|gz") as inner:
                processed = 0
                for member in inner:
                    match = VIDEO_NAME.fullmatch(member.name)
                    if match is None:
                        continue
                    name = f"dia{match.group(1)}_utt{match.group(2)}.mp4"
                    if name not in selected[split]:
                        continue
                    row = labels[split][name]
                    stream = inner.extractfile(member)
                    if stream is None:
                        continue
                    video_bytes = stream.read()
                    label = LABELS[row["Emotion"]]
                    speaker = row["Speaker"]
                    future = pool.submit(process_video, video_bytes, temporary_output, split, label, Path(name).stem)
                    pending.add((future, split, label, speaker))
                    processed += 1
                    if len(pending) >= args.workers * 3:
                        done, _ = wait([x[0] for x in pending], return_when=FIRST_COMPLETED)
                        complete = [x for x in pending if x[0] in done]
                        for fut, s, lbl, spk in complete:
                            result, n = fut.result()
                            counts[f"{s}/{lbl}/{result}/videos"] += 1
                            counts[f"{s}/{lbl}/frames"] += n
                            counts[f"{s}/speaker/{spk}/frames"] += n
                            pending.remove((fut,s,lbl,spk))
                    if processed % 250 == 0:
                        print(split, "videos_seen", processed, "saved_frames", sum(
                            v for k,v in counts.items() if k.startswith(split+"/") and k.endswith("/frames")
                            and "/speaker/" not in k), flush=True)
            print("archive_split_read", split, processed, flush=True)
    for fut, split, label, speaker in pending:
        result, n = fut.result()
        counts[f"{split}/{label}/{result}/videos"] += 1
        counts[f"{split}/{label}/frames"] += n
        counts[f"{split}/speaker/{speaker}/frames"] += n
    report = {
        "source": str(args.archive),
        "source_sha256": "a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e",
        "source_authors": "https://github.com/declare-lab/MELD",
        "label_warning": "Utterance labels may depend on speech/context. A unique visible face is not guaranteed to be the speaking character.",
        "sampling": "Deterministic hash cap by class; single high-confidence face only; video held within the official dialogue split.",
        "identity_warning": "The six principal actors recur in all three official splits. Use other datasets for out-of-domain testing.",
        "counts": dict(sorted(counts.items())),
    }
    (temporary_output / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_output.replace(args.output)
    print("dataset_ready", args.output, flush=True)


if __name__ == "__main__":
    main()
