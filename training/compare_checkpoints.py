from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from train_transformer import CLASSES, build_transforms, load_split, metrics_from_confusion
from emotion_preprocessing import EmotionEvaluationTransform


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare compatible checkpoints on fixed image-folder datasets")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", nargs=2, action="append", metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--inference-precision", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports: dict[str, object] = {}
    for candidate in args.candidates:
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        if tuple(checkpoint['classes']) != CLASSES:
            raise ValueError('Candidate class order differs from the seven-class contract')
        transform = EmotionEvaluationTransform(int(checkpoint['image_size']), grayscale=checkpoint.get('grayscale', True), evaluation_zoom=args.zoom)
        datasets = {name: load_split(Path(path), transform, allow_empty=True) for name, path in args.dataset}
        loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                   for name, dataset in datasets.items()}
        model = timm.create_model(
            checkpoint["model_name"], pretrained=False, num_classes=len(CLASSES),
            **checkpoint.get('model_kwargs', {})
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        candidate_report = {}
        with torch.inference_mode():
            for name, loader in loaders.items():
                confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
                records = []; offset = 0
                for images, targets in loader:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and args.inference_precision == "bfloat16",
                    ):
                        logits = model(images.to(device))
                    predictions = logits.argmax(dim=1).cpu()
                    if args.save_predictions:
                        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
                        for i, (target, prediction, scores) in enumerate(zip(targets, predictions, probabilities)):
                            records.append({'file':str(datasets[name].samples[offset+i][0]), 'label':CLASSES[int(target)],
                                            'predicted':CLASSES[int(prediction)], 'probabilities':scores.tolist()})
                    offset += len(targets)
                    confusion += torch.bincount(
                        targets * len(CLASSES) + predictions,
                        minlength=len(CLASSES) ** 2,
                    ).reshape(len(CLASSES), len(CLASSES))
                metrics = metrics_from_confusion(confusion, loss=0.0)
                candidate_report[name] = {
                    "examples": len(datasets[name]),
                    "accuracy": metrics.accuracy,
                    "macro_f1": metrics.macro_f1,
                    "per_class": metrics.per_class,
                    "confusion": confusion.tolist(),
                }
                if args.save_predictions:
                    candidate_report[name]['records'] = records
        reports[str(candidate)] = candidate_report
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        compact = {
            name: {
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "anger": values["per_class"]["anger"],
                "fear": values["per_class"]["fear"],
            }
            for name, values in candidate_report.items()
        }
        print(json.dumps({candidate.name: compact}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"device": str(device), "inference_precision": args.inference_precision if device.type == "cuda" else "float32", "evaluation_zoom": args.zoom, "reports": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
