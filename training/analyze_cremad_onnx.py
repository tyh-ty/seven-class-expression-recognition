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
    r"^cremad_actor(?P<actor>\d+)_(?P<source_actor>\d+)_"
    r"(?P<sentence>[A-Z]+)_(?P<emotion>[A-Z]+)_(?P<intensity>[A-Z]+)_f(?P<frame>\d+)\.jpg$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CREMA-D face frames and mean-logit video predictions"
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
    supported = np.bincount(targets, minlength=len(CLASSES)) > 0
    per_class_f1 = [metrics.per_class[label]["f1"] for label in CLASSES]
    return {
        "examples": int(targets.size),
        "accuracy": metrics.accuracy,
        "macro_f1_supported": float(np.mean(np.asarray(per_class_f1)[supported])),
        "balanced_accuracy_supported": float(
            np.mean([metrics.per_class[label]["recall"] for i, label in enumerate(CLASSES) if supported[i]])
        ),
        "per_class": metrics.per_class,
        "confusion": confusion.tolist(),
    }


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
                raise ValueError(f"Unexpected CREMA-D filename: {path.name}")
            metadata = match.groupdict()
            records.append(
                {
                    "path": path,
                    "target": target,
                    "actor": int(metadata["actor"]),
                    "frame": int(metadata["frame"]),
                    "intensity": metadata["intensity"],
                    "video_id": path.name.rsplit("_f", 1)[0],
                }
            )
    if not records:
        raise RuntimeError(f"No CREMA-D face images found in {args.data}")

    logits_parts: list[np.ndarray] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        inputs = np.stack([emotion_tensor_path(record["path"]) for record in batch])
        logits_parts.append(session.run(None, {input_name: inputs})[0])
    logits = np.concatenate(logits_parts)
    targets = np.asarray([record["target"] for record in records], dtype=np.int64)
    predictions = logits.argmax(axis=1)

    videos: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        videos[str(record["video_id"])].append(index)
    video_targets: list[int] = []
    video_predictions: list[int] = []
    for indices in videos.values():
        video_targets.append(int(records[indices[0]]["target"]))
        video_predictions.append(int(logits[indices].mean(axis=0).argmax()))

    report = {
        "model": str(args.model),
        "data": str(args.data),
        "frame_level": metric_report(targets, predictions),
        "video_level_mean_logits": metric_report(
            np.asarray(video_targets), np.asarray(video_predictions)
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
