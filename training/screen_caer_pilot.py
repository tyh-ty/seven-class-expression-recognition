"""Screen CAER video candidates for manual facial-label review, never training.

Use the camera's 512-pixel/JPEG-85 input convention and the product YuNet detector.
Scene labels are retained only as unverified source labels. No classifier score
selects frames, and no source test videos or random frame splits are permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import (  # noqa: E402
    EMOTIONS,
    FACE_DETECTION_SCORE_THRESHOLD,
    FACE_DETECTOR_PATH,
    EmotionRecognizer,
)

LABELS = tuple(label for label, _ in EMOTIONS)
FRACTIONS = (0.25, 0.50, 0.75)
REVIEW_STATUS = "pending_manual_facial_label_review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_detector() -> Any:
    """Load only the product's small face detector, without emotion weights."""
    model_path = EmotionRecognizer._opencv_safe_path(FACE_DETECTOR_PATH)
    return cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320), FACE_DETECTION_SCORE_THRESHOLD, 0.3, 5000
    )


def camera_frame(frame: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Emulate browser canvas dimensions and JPEG quality; do not upscale."""
    height, width = frame.shape[:2]
    if max_side <= 0:
        raise ValueError('Capture size must be positive')
    scale = min(1.0, max_side / max(width, height))
    size = (max(1, math.floor(width * scale + 0.5)),
            max(1, math.floor(height * scale + 0.5)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Cannot encode camera JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Cannot decode camera JPEG")
    return decoded


def detect_faces(detector: Any, image: np.ndarray) -> list:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, raw_faces = detector.detect(image)
    candidates = [] if raw_faces is None else [
        (int(round(face[0])), int(round(face[1])),
         int(round(face[2])), int(round(face[3])), float(face[14]),
         np.asarray(face[4:14], dtype=np.float32).reshape(5, 2))
        for face in raw_faces if face[2] >= 24 and face[3] >= 24
    ]
    return EmotionRecognizer._deduplicate_faces(candidates)


def screen_video(record: dict, video_path: Path, detector: Any,
                 crop_root: Path, minimum_face: int = 55) -> dict:
    """Save eligible crops, preserving the source clip and split of every frame."""
    result = {key: record[key] for key in
              ("source_entry", "split", "label", "file", "sha256")}
    result.update(status="decode_failed", review_status=REVIEW_STATUS,
                  usable_for_training=False, frames=[])
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            result["error"] = "VideoCapture could not open video"
            return result
        frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(value) for value in
                   (frame_count_value, fps, width, height)):
            result["error"] = "Non-finite video metadata"
            return result
        frame_count = int(frame_count_value)
        result.update(source_width=int(width), source_height=int(height),
                      source_fps=float(fps), source_frame_count=frame_count)
        if frame_count <= 0 or width <= 0 or height <= 0:
            result["error"] = "Invalid video metadata"
            return result
        used_indices: set[int] = set()
        for fraction in FRACTIONS:
            index = min(frame_count - 1, int((frame_count - 1) * fraction + 0.5))
            item: dict = {"fraction": fraction, "frame_index": index,
                          "source_timestamp_seconds": index / fps if fps > 0 else None,
                          "status": "decode_failed"}
            if index in used_indices:
                item["status"] = "duplicate_sample_index"
                result["frames"].append(item)
                continue
            used_indices.add(index)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                item["error"] = "Cannot seek to requested frame"
                result["frames"].append(item)
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                result["frames"].append(item)
                continue
            image = camera_frame(frame)
            image_height, image_width = image.shape[:2]
            faces = detect_faces(detector, image)
            item.update(capture_width=image_width, capture_height=image_height,
                        face_count=len(faces))
            if not faces:
                item["status"] = "no_face"
            elif len(faces) > 1:
                item["status"] = "multiple_faces"
            else:
                x, y, face_width, face_height, confidence, _ = faces[0]
                item.update(face_box=[x, y, face_width, face_height],
                            detection_confidence=confidence)
                if min(face_width, face_height) < minimum_face:
                    item["status"] = "single_face_too_small"
                else:
                    left, top, crop_width, crop_height = EmotionRecognizer._square_face_box(
                        x, y, face_width, face_height, image_width, image_height
                    )
                    crop = image[top:top + crop_height, left:left + crop_width]
                    if crop.size == 0:
                        item["status"] = "empty_crop"
                    else:
                        clip_id = Path(record["source_entry"]).stem
                        destination = crop_root / record["split"] / record["label"] / (
                            f"caer_{clip_id}-{record['sha256'][:10]}_f{index:06d}.jpg"
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        ok, encoded = cv2.imencode(
                            ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )
                        if not ok:
                            raise RuntimeError("Cannot encode review crop")
                        destination.write_bytes(encoded.tobytes())
                        item.update(status="review_candidate",
                                    crop_box=[left, top, crop_width, crop_height],
                                    crop_file=destination.relative_to(crop_root).as_posix(),
                                    review_status=REVIEW_STATUS,
                                    usable_for_training=False)
            result["frames"].append(item)
        decoded = [frame for frame in result["frames"]
                   if frame["status"] not in ("decode_failed", "duplicate_sample_index")]
        result["status"] = "screened" if decoded else "decode_failed"
        result["decoded_sample_frames"] = len(decoded)
        return result
    finally:
        capture.release()


def prepare_records(source_manifest: dict, source_root: Path) -> tuple[list[dict], list[dict]]:
    """Verify hashes first so all cross-split duplicate clips can be excluded."""
    records = []
    exclusions = []
    by_hash: dict[str, list[dict]] = defaultdict(list)
    seen_entries = set()
    for original in sorted(source_manifest["videos"], key=lambda row: row["source_entry"]):
        record = dict(original)
        source_entry = record.get("source_entry", "")
        split = record.get("split")
        expected_source_split = {"train": "train", "val": "validation"}.get(split)
        if (expected_source_split is None or record.get("label") not in LABELS
                or source_entry.split("/")[0] != expected_source_split):
            raise ValueError(f"Only source train/validation clips are allowed: {source_entry}")
        if source_entry in seen_entries:
            exclusions.append(dict(record, exclusion="duplicate_manifest_entry"))
            continue
        seen_entries.add(source_entry)
        video_path = (source_root / record["file"]).resolve()
        if not video_path.is_relative_to(source_root.resolve()):
            raise ValueError(f"Video path outside source directory: {source_entry}")
        if not video_path.is_file():
            exclusions.append(dict(record, exclusion="missing_video"))
            continue
        actual_hash = sha256_file(video_path)
        if actual_hash != record.get("sha256"):
            exclusions.append(dict(record, exclusion="sha256_mismatch", actual_sha256=actual_hash))
            continue
        record["sha256"] = actual_hash
        by_hash[actual_hash].append(record)
    for digest, group in sorted(by_hash.items()):
        if len({record["split"] for record in group}) > 1:
            exclusions.extend(dict(record, exclusion="cross_split_duplicate_video") for record in group)
        elif len({record["label"] for record in group}) > 1:
            exclusions.extend(dict(record, exclusion="duplicate_video_conflicting_labels") for record in group)
        else:
            records.append(group[0])
            exclusions.extend(dict(record, exclusion="same_split_duplicate_video",
                                   kept_source_entry=group[0]["source_entry"]) for record in group[1:])
    return sorted(records, key=lambda row: row["source_entry"]), exclusions


def contact_sheets(videos: list[dict], crop_root: Path, report_root: Path) -> dict:
    sheets = {}
    for label in LABELS:
        candidates = [(video, frame) for video in videos
                      if video["split"] == "train" and video["label"] == label
                      for frame in video["frames"] if frame["status"] == "review_candidate"]
        candidates.sort(key=lambda pair: hashlib.sha256(
            ("caer-review-v1:" + pair[0]["source_entry"] + ":" +
             str(pair[1]["frame_index"])).encode("utf-8")
        ).hexdigest())
        chosen = candidates[:16]
        canvas = Image.new("RGB", (4 * 208, 62 + 4 * 238), "#f0f1f3")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"CAER {label}: unreviewed source scene labels", fill="#16181d")
        draw.text((12, 28), "Manual face-label review required. NOT training ground truth.", fill="#b03325")
        draw.text((12, 44), "Fixed hash order; no model-score selection.", fill="#555555")
        selected = []
        for index, (video, frame) in enumerate(chosen):
            x, y = (index % 4) * 208, 62 + (index // 4) * 238
            with Image.open(crop_root / frame["crop_file"]) as image:
                tile = ImageOps.contain(image.convert("RGB"), (192, 192))
                canvas.paste(tile, (x + 8 + (192 - tile.width) // 2,
                                    y + (192 - tile.height) // 2))
            caption = f"{Path(video['source_entry']).stem} frame {frame['frame_index']}"
            draw.text((x + 8, y + 195), caption, fill="#16181d")
            draw.text((x + 8, y + 212), "label: PENDING REVIEW", fill="#b03325")
            selected.append({"source_entry": video["source_entry"],
                             "frame_index": frame["frame_index"],
                             "crop_file": frame["crop_file"]})
        path = report_root / f"train-{label}-contact.jpg"
        canvas.save(path, quality=90)
        sheets[label] = {"file": str(path), "eligible_train_frames": len(candidates),
                         "selected": selected}
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(r"D:\CAER-source\pilot-v1\manifest.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/caer/pilot-review-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "output/caer-pilot-review-20260927")
    parser.add_argument("--minimum-face", type=int, default=55,
                        help="Minimum detected width AND height in the captured 512px frame")
    args = parser.parse_args()
    if args.minimum_face < 24:
        parser.error("minimum face must be at least the product detector filter (24px)")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, exclusions = prepare_records(manifest, args.manifest.parent)
    # Use a new review directory per run. This prevents stale previously-eligible
    # crops from surviving when a new manifest exposes cross-split duplicates.
    if args.output.exists() and any(args.output.rglob("*.jpg")):
        parser.error("Review output already contains crops; choose a new --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    detector = create_detector()
    videos = []
    for index, record in enumerate(records, 1):
        result = screen_video(record, args.manifest.parent / record["file"], detector,
                              args.output, args.minimum_face)
        videos.append(result)
        if index % 20 == 0 or index == len(records):
            print(f"screened={index}/{len(records)} review_frames=" +
                  str(sum(frame["status"] == "review_candidate"
                          for video in videos for frame in video["frames"])), flush=True)
    frame_counts = Counter(frame["status"] for video in videos for frame in video["frames"])
    face_counts = Counter("zero" if frame["face_count"] == 0 else
                          "single" if frame["face_count"] == 1 else "multiple"
                          for video in videos for frame in video["frames"] if "face_count" in frame)
    by_class = {}
    for split in ("train", "val"):
        for label in LABELS:
            selected = [video for video in videos if video["split"] == split and video["label"] == label]
            by_class[f"{split}/{label}"] = {
                "videos": len(selected),
                "frames": dict(Counter(frame["status"] for video in selected for frame in video["frames"]))}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "review_status": REVIEW_STATUS, "usable_for_training": False,
        "trained": False,
        "warning": "Source labels describe movie/TV scenes. These facial crops are NOT verified "
                   "ground truth and MUST NOT be fed directly into training or accuracy evaluation.",
        "split_policy": "Source train and validation retained. No test data read; no random frame split. "
                        "All exact video duplicates spanning splits excluded. Identity-disjointness unknown.",
        "preprocessing": {"capture_max_side": 512, "jpeg_quality": 85,
                          "resize": "OpenCV bilinear emulation of browser canvas dimensions",
                          "detector": FACE_DETECTOR_PATH.name,
                          "detector_threshold": FACE_DETECTION_SCORE_THRESHOLD,
                          "deduplication": "server.EmotionRecognizer._deduplicate_faces",
                          "required_face_count": 1, "minimum_face_width_and_height": args.minimum_face,
                          "crop": "server.EmotionRecognizer._square_face_box (1.45)",
                          "crop_storage_jpeg_quality": 95, "sample_fractions": list(FRACTIONS),
                          "emotion_classifier_used": False},
        "source_video_count": len(manifest["videos"]),
        "screened_video_count": len(videos),
        "frame_status_counts": dict(frame_counts), "face_count_categories": dict(face_counts),
        "per_split_label": by_class,
        "exclusions": exclusions,
        "exclusion_counts": dict(Counter(row["exclusion"] for row in exclusions)),
        "videos": videos,
    }
    report["contact_sheets"] = contact_sheets(videos, args.output, args.report)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output / "manifest.json").write_text(text, encoding="utf-8")
    (args.report / "screening.json").write_text(text, encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "screened_video_count", "frame_status_counts", "face_count_categories", "exclusion_counts"
    )}), flush=True)


if __name__ == "__main__":
    main()
