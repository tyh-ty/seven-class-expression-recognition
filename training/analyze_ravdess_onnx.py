from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from emotion_preprocessing import emotion_tensor_path
from train_transformer import CLASSES, metrics_from_confusion


FILENAME_PATTERN = re.compile(
    r"^actor(?P<actor>\d+)_"
    r"(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<source_actor>\d{2})_f(?P<frame>\d+)\.jpg$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Break down RAVDESS ONNX performance by actor, intensity, and frame position"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metric_report(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    metrics = metrics_from_confusion(torch.from_numpy(confusion), loss=0.0)
    return {
        "examples": int(targets.size),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "per_class": metrics.per_class,
        "confusion": confusion.tolist(),
    }


def grouped_reports(
    values: list[str | int], targets: np.ndarray, predictions: np.ndarray
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    array = np.asarray(values)
    for value in sorted(set(values), key=str):
        selected = array == value
        result[str(value)] = metric_report(targets[selected], predictions[selected])
    return result


def main() -> None:
    args = parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    records: list[dict[str, object]] = []
    for target, label in enumerate(CLASSES):
        for path in sorted((args.data / label).glob("*.jpg")):
            match = FILENAME_PATTERN.match(path.name)
            if match is None:
                raise ValueError(f"Unexpected RAVDESS filename: {path.name}")
            metadata = {key: int(value) for key, value in match.groupdict().items()}
            video_id = path.name.rsplit("_f", 1)[0]
            records.append(
                {
                    "path": path,
                    "target": target,
                    "video_id": video_id,
                    **metadata,
                }
            )
    if not records:
        raise RuntimeError(f"No RAVDESS face images found in {args.data}")

    all_logits: list[np.ndarray] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        inputs = np.stack([emotion_tensor_path(record["path"]) for record in batch])
        all_logits.append(session.run(None, {input_name: inputs})[0])
    logits = np.concatenate(all_logits)
    targets = np.asarray([record["target"] for record in records], dtype=np.int64)
    predictions = logits.argmax(axis=1)

    videos: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        videos[str(record["video_id"])].append(index)
    video_targets: list[int] = []
    video_predictions: list[int] = []
    video_actors: list[int] = []
    video_intensities: list[int] = []
    for indices in videos.values():
        first = records[indices[0]]
        video_targets.append(int(first["target"]))
        video_predictions.append(int(logits[indices].mean(axis=0).argmax()))
        video_actors.append(int(first["actor"]))
        video_intensities.append(int(first["intensity"]))

    margins: dict[str, dict[str, float]] = {}
    for label in ("anger", "disgust"):
        selected = targets == CLASSES.index(label)
        anger_disgust_margin = logits[selected, CLASSES.index("anger")] - logits[
            selected, CLASSES.index("disgust")
        ]
        margins[label] = {
            "mean_anger_minus_disgust_logit": round(float(anger_disgust_margin.mean()), 6),
            "median_anger_minus_disgust_logit": round(float(np.median(anger_disgust_margin)), 6),
            "anger_logit_wins_fraction": round(float((anger_disgust_margin > 0).mean()), 6),
        }

    report = {
        "model": str(args.model),
        "data": str(args.data),
        "frame_level": metric_report(targets, predictions),
        "by_frame_index": grouped_reports(
            [int(record["frame"]) for record in records], targets, predictions
        ),
        "by_actor": grouped_reports(
            [int(record["actor"]) for record in records], targets, predictions
        ),
        "by_intensity": grouped_reports(
            [int(record["intensity"]) for record in records], targets, predictions
        ),
        "anger_disgust_margins": margins,
        "video_level_mean_logits": metric_report(
            np.asarray(video_targets), np.asarray(video_predictions)
        ),
        "video_by_actor": grouped_reports(
            video_actors, np.asarray(video_targets), np.asarray(video_predictions)
        ),
        "video_by_intensity": grouped_reports(
            video_intensities, np.asarray(video_targets), np.asarray(video_predictions)
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
