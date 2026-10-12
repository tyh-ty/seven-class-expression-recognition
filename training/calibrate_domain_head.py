from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader

from train_transformer import CLASSES, build_transforms, load_split


def scores(targets: np.ndarray, logits: np.ndarray) -> dict:
    matrix = np.bincount(
        targets * len(CLASSES) + logits.argmax(1), minlength=len(CLASSES)**2
    ).reshape(len(CLASSES), len(CLASSES))
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    tp = matrix.diagonal()
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    recall = tp / np.maximum(support, 1)
    valid = support > 0
    return {
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_f1": float(f1[valid].mean()),
        "per_class": {
            label: {"support": int(support[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(CLASSES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit small classifier bias corrections on validation identities only")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--ferplus", type=Path, required=True)
    parser.add_argument("--ravdess", type=Path, required=True)
    parser.add_argument("--cremad", type=Path, required=True)
    parser.add_argument("--kdef", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform = build_transforms(224)
    datasets = {
        name: load_split(path, transform, allow_empty=True)
        for name, path in (("ferplus", args.ferplus), ("ravdess", args.ravdess),
                           ("cremad", args.cremad), ("kdef", args.kdef))
    }
    targets = {name: np.asarray(dataset.targets) for name, dataset in datasets.items()}
    args.output.mkdir(parents=True, exist_ok=True)

    def predict(path: Path) -> dict[str, np.ndarray]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Checkpoint classes differ")
        model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(CLASSES)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        result = {}
        with torch.inference_mode():
            for name, dataset in datasets.items():
                batches = []
                loader = DataLoader(dataset, batch_size=64, num_workers=0)
                for images, _ in loader:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits = model(images.to(device))
                    batches.append(logits.float().cpu().numpy())
                result[name] = np.concatenate(batches)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"validation predictions complete: {path}", flush=True)
        return result

    base_logits = predict(args.base)
    baseline = {n: scores(targets[n], z) for n, z in base_logits.items()}

    def assess(logits: dict[str, np.ndarray], bias: np.ndarray) -> tuple[float, dict]:
        metrics = {n: scores(targets[n], z + bias) for n, z in logits.items()}
        violations = max(0, baseline["ferplus"]["macro_f1"] - .006 - metrics["ferplus"]["macro_f1"])
        violations += max(0, baseline["ferplus"]["accuracy"] - .004 - metrics["ferplus"]["accuracy"])
        violations += max(0, baseline["ravdess"]["macro_f1"] - .01 - metrics["ravdess"]["macro_f1"])
        # Give every source equal influence and penalize large decision shifts.
        value = np.mean([m["macro_f1"] for m in metrics.values()])
        value -= 30 * violations + .001 * np.square(bias).mean()
        return float(value), metrics

    records = []
    best = None
    for path in args.candidates:
        logits = predict(path)
        bias = np.zeros(len(CLASSES), dtype=np.float32)
        value, metrics = assess(logits, bias)
        before = metrics
        for step in (.3, .15, .075):
            for _ in range(4):
                changed = False
                for i in range(len(CLASSES)):
                    for shift in (-step, step):
                        proposed = bias.copy()
                        proposed[i] = np.clip(proposed[i] + shift, -.6, .6)
                        proposed_value, proposed_metrics = assess(logits, proposed)
                        if proposed_value > value + 1e-8:
                            bias, value, metrics = proposed, proposed_value, proposed_metrics
                            changed = True
                if not changed:
                    break
        feasible = (
            metrics["ferplus"]["macro_f1"] >= baseline["ferplus"]["macro_f1"] - .006
            and metrics["ferplus"]["accuracy"] >= baseline["ferplus"]["accuracy"] - .004
            and metrics["ravdess"]["macro_f1"] >= baseline["ravdess"]["macro_f1"] - .01
        )
        record = {"checkpoint": str(path), "bias": bias.tolist(), "score": value,
                  "feasible": feasible, "before": before, "after": metrics}
        records.append(record)
        if feasible and (best is None or value > best["score"]):
            best = record
        print(f"candidate={path.name} score={value:.6f} feasible={feasible} bias={bias.tolist()}", flush=True)

    report = {"method": "validation-only bounded classifier bias calibration; equal source macro-F1",
              "validation_paths": {n: str(dataset.root) for n, dataset in datasets.items()},
              "baseline": baseline, "candidates": records, "selected": best,
              "test_data_used_for_calibration": False}
    (args.output / "validation_selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if best is None:
        raise RuntimeError("No candidate satisfied validation retention constraints")
    checkpoint = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["model_state"]["head.bias"] += torch.tensor(best["bias"])
    checkpoint["validation"] = best["after"]["ferplus"]
    checkpoint["domain_calibration"] = best
    checkpoint["epoch"] = "validation-calibrated"
    checkpoint.pop("target_validation", None)
    torch.save(checkpoint, args.output / "selected.pt")
    print(f"selected: {best['checkpoint']} -> {args.output / 'selected.pt'}")


if __name__ == "__main__":
    main()
