from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from train_transformer import (
    CLASSES,
    build_transforms,
    metrics_from_confusion,
    supported_metrics_from_confusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an emotion ONNX model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, evaluation_transform = build_transforms(args.image_size, grayscale=args.grayscale)
    dataset = datasets.ImageFolder(
        args.data, transform=evaluation_transform, allow_empty=True
    )
    if tuple(dataset.classes) != CLASSES:
        raise ValueError(f"Expected classes {CLASSES}, found {tuple(dataset.classes)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    inference_seconds = 0.0
    records = []
    offset = 0

    for images, targets in loader:
        started_at = perf_counter()
        logits = session.run(None, {input_name: images.numpy()})[0]
        if args.horizontal_flip_tta:
            flipped_logits = session.run(
                None,
                {input_name: torch.flip(images, dims=(3,)).numpy()},
            )[0]
            logits = (logits + flipped_logits) / 2.0
        inference_seconds += perf_counter() - started_at
        predictions = torch.from_numpy(np.asarray(logits)).argmax(dim=1)
        if args.save_predictions:
            probabilities = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=1).numpy()
            for index, (target, predicted, scores) in enumerate(zip(targets, predictions, probabilities)):
                records.append({'file': str(dataset.samples[offset + index][0]),
                                'label': CLASSES[int(target)], 'predicted': CLASSES[int(predicted)],
                                'probabilities': scores.tolist()})
        offset += len(targets)
        confusion += torch.bincount(
            targets * len(CLASSES) + predictions,
            minlength=len(CLASSES) ** 2,
        ).reshape(len(CLASSES), len(CLASSES))

    metrics = metrics_from_confusion(confusion, loss=0.0)
    report = {
        "model": str(args.model),
        "examples": len(dataset),
        "metrics": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "balanced_accuracy": metrics.balanced_accuracy,
            "per_class": metrics.per_class,
        },
        "supported_metrics": supported_metrics_from_confusion(confusion),
        "confusion": confusion.tolist(),
        "inference_seconds": round(inference_seconds, 3),
        "milliseconds_per_image": round(inference_seconds * 1000 / len(dataset), 3),
        "horizontal_flip_tta": args.horizontal_flip_tta,
        "grayscale": args.grayscale,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "latency_scope": "Batch throughput divided by image count; not single-face interactive latency.",
    }
    if args.save_predictions:
        report['records'] = records
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
