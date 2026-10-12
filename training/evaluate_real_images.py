from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from curate_web_emotions import CLASSES, IMAGE_SUFFIXES, model_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an emotion ONNX model on a possibly partial class folder"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metrics(confusion: np.ndarray, minimum_support: int = 5) -> dict[str, object]:
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = true_positive / np.maximum(predicted, 1)
    recall = true_positive / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    supported = support > 0
    selected = support >= minimum_support
    if not selected.any():
        selected = supported
    total = max(float(confusion.sum()), 1.0)
    return {
        "accuracy": round(float(true_positive.sum() / total), 6),
        "macro_f1_supported": round(float(f1[supported].mean()), 6),
        "balanced_accuracy_supported": round(float(recall[supported].mean()), 6),
        "supported_classes": [
            label for index, label in enumerate(CLASSES) if supported[index]
        ],
        "macro_f1_min_support": round(float(f1[selected].mean()), 6),
        "balanced_accuracy_min_support": round(float(recall[selected].mean()), 6),
        "minimum_support": minimum_support,
        "selected_classes": [
            label for index, label in enumerate(CLASSES) if selected[index]
        ],
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASSES)
        },
    }


def main() -> None:
    args = parse_args()
    samples: list[tuple[Path, int]] = []
    for class_index, label in enumerate(CLASSES):
        class_root = args.data / label
        if not class_root.is_dir():
            continue
        for path in sorted(class_root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, class_index))
    if not samples:
        raise ValueError(f"No images found under {args.data}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    started_at = perf_counter()
    for start in range(0, len(samples), args.batch_size):
        batch_samples = samples[start : start + args.batch_size]
        batch = np.stack([model_input(path) for path, _ in batch_samples])
        logits = np.asarray(session.run(None, {input_name: batch})[0])
        predictions = logits.argmax(axis=1)
        for (_, target), prediction in zip(batch_samples, predictions, strict=True):
            confusion[target, int(prediction)] += 1
    duration = perf_counter() - started_at
    report = {
        "model": str(args.model),
        "data": str(args.data),
        "examples": len(samples),
        "metrics": metrics(confusion),
        "confusion": confusion.tolist(),
        "inference_seconds": round(duration, 3),
        "milliseconds_per_image": round(duration * 1000 / len(samples), 3),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
