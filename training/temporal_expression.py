"""Evaluate a lightweight two-frame adapter without changing the deployed model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from emotion_preprocessing import emotion_tensor_path
from train_transformer import CLASSES, metrics_from_confusion


INTENDED_LABELS = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happiness", "NEU": "neutral", "SAD": "sadness",
}
VOTE_LABELS = {
    "A": "anger", "D": "disgust", "F": "fear",
    "H": "happiness", "N": "neutral", "S": "sadness",
}


@dataclass(frozen=True)
class Pair:
    first: Path
    second: Path
    label: int
    source: str
    level: str
    video: str


class PairDataset(Dataset):
    def __init__(self, records: list[Pair]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        return (
            torch.from_numpy(emotion_tensor_path(item.first)),
            torch.from_numpy(emotion_tensor_path(item.second)),
            index,
        )


def read_votes(path: Path) -> dict[str, tuple[str | None, float]]:
    votes = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            try:
                confidence = float(row["FaceLevel"])
            except (TypeError, ValueError):
                confidence = -1.0
            votes[row["FileName"].strip()] = (
                VOTE_LABELS.get(row["FaceVote"].strip()), confidence
            )
    return votes


def video_pairs(root: Path, split: str, source: str, votes: dict, min_confidence: float) -> list[Pair]:
    groups: dict[tuple[int, str], dict[int, Path]] = {}
    levels: dict[tuple[int, str], str] = {}
    for label, name in enumerate(CLASSES):
        for path in sorted((root / split / name).glob("*_f*.jpg")):
            try:
                stem, index_text = path.stem.rsplit("_f", 1)
                frame = int(index_text)
            except ValueError:
                continue
            if frame not in (0, 1):
                continue
            if source == "ravdess":
                encoded = stem.split("_", 1)[1]
                fields = encoded.split("-")
                level = fields[3]
                if level != "01":
                    continue
            else:
                encoded = stem.split("_", 2)[2]
                fields = encoded.split("_")
                level = fields[-1]
                intended = INTENDED_LABELS.get(fields[-2])
                voted, confidence = votes.get(encoded, (None, -1))
                # Exclude uncertain or conflicting visual labels. XX is an
                # unspecified acting intensity, not a neutral emotion label.
                if voted != name or intended != name or confidence < min_confidence:
                    continue
                if level not in {"LO", "MD", "XX"}:
                    continue
            key = (label, stem)
            groups.setdefault(key, {})[frame] = path
            levels[key] = level
    result = []
    for (label, video), frames in sorted(groups.items()):
        if 0 in frames and 1 in frames:
            result.append(Pair(frames[0], frames[1], label, source, levels[(label, video)], video))
    return result


def collect(args: argparse.Namespace, split: str) -> list[Pair]:
    ravdess_root = args.ravdess_test if split == "test" else args.ravdess
    votes = read_votes(args.summary)
    records = video_pairs(ravdess_root, split, "ravdess", votes, args.min_confidence)
    records.extend(video_pairs(args.cremad, split, "cremad", votes, args.min_confidence))
    counts = Counter(f"{p.source}/{p.level}/{CLASSES[p.label]}" for p in records)
    print(split, len(records), dict(sorted(counts.items())), flush=True)
    if not records:
        raise RuntimeError(f"No video pairs in {split}")
    return records


def extract(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        checkpoint["model_name"], pretrained=False, num_classes=len(CLASSES)
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        target = args.output / f"features-{split}.npz"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        records = collect(args, split)
        loader = DataLoader(
            PairDataset(records), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        logits = np.zeros((len(records), 2, len(CLASSES)), dtype=np.float32)
        delta = np.zeros((len(records), 384), dtype=np.float32)
        with torch.inference_mode():
            for batch_index, (first, second, indices) in enumerate(loader):
                images = torch.cat((first, second), dim=0).to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    tokens = model.forward_features(images)
                    embeddings = model.forward_head(tokens, pre_logits=True)
                    predictions = model.get_classifier()(embeddings)
                n = len(indices)
                output = predictions.float().cpu().numpy()
                vector = embeddings.float().cpu().numpy()
                positions = indices.numpy()
                logits[positions, 0] = output[:n]
                logits[positions, 1] = output[n:]
                delta[positions] = vector[n:] - vector[:n]
                if (batch_index + 1) % 30 == 0:
                    print(split, f"{(batch_index+1)*args.batch_size}/{len(records)}", flush=True)
        np.savez_compressed(
            target, logits=logits, delta=delta,
            label=np.asarray([p.label for p in records], dtype=np.int64),
            source=np.asarray([p.source for p in records]),
            level=np.asarray([p.level for p in records]),
            video=np.asarray([p.video for p in records]),
        )
        print("saved", target, flush=True)


def report(logits: np.ndarray, labels: np.ndarray) -> dict:
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    metrics = metrics_from_confusion(torch.from_numpy(confusion), loss=0)
    supported = [i for i in range(len(CLASSES)) if confusion[i].sum() > 0]
    return {
        "examples": len(labels), "accuracy": metrics.accuracy,
        "macro_f1_supported": round(
            sum(metrics.per_class[CLASSES[i]]["f1"] for i in supported) / len(supported), 6
        ),
        "per_class": metrics.per_class, "confusion": confusion.tolist(),
    }


def group_reports(logits: np.ndarray, data) -> dict:
    masks = {"all": np.ones(len(data["label"]), dtype=bool)}
    source = data["source"]
    level = data["level"]
    masks["ravdess_normal_early"] = source == "ravdess"
    masks["cremad_low_medium"] = (source == "cremad") & np.isin(level, ["LO", "MD"])
    masks["cremad_unknown_intensity"] = (source == "cremad") & (level == "XX")
    return {key: report(logits[mask], data["label"][mask]) for key, mask in masks.items() if mask.any()}


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    arrays = {split: np.load(args.output / f"features-{split}.npz") for split in ("train", "val", "test")}
    baseline = {s: a["logits"].mean(axis=1) for s, a in arrays.items()}
    # Time-aware input: first/last evidence and changes in ViT class logits
    # and 384-dimensional ViT face embeddings. Inference reuses the existing ViT.
    raw = {
        s: np.concatenate((a["logits"][:, 0], a["logits"][:, 1],
                           a["logits"][:, 1] - a["logits"][:, 0], a["delta"]), axis=1)
        for s, a in arrays.items()
    }
    mean = raw["train"].mean(axis=0)
    std = np.maximum(raw["train"].std(axis=0), 0.05)
    features = {s: torch.from_numpy((a - mean) / std).float() for s, a in raw.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = features["train"].to(device)
    train_baseline = torch.from_numpy(baseline["train"]).to(device)
    targets = torch.from_numpy(arrays["train"]["label"]).to(device)
    class_counts = torch.bincount(targets, minlength=len(CLASSES)).float()
    weights = (class_counts.mean() / class_counts).sqrt()
    weights /= weights.mean()
    print("train_counts", class_counts.tolist(), "weights", weights.tolist(), flush=True)

    baselines = {s: group_reports(baseline[s], arrays[s]) for s in ("val", "test")}
    print("baseline_val", json.dumps(baselines["val"], ensure_ascii=False)[:1000], flush=True)
    best = None
    for kind in ("logits", "embeddings"):
        size = 21 if kind == "logits" else raw["train"].shape[1]
        correction = nn.Linear(size, len(CLASSES)).to(device)
        nn.init.zeros_(correction.weight)
        nn.init.zeros_(correction.bias)
        optimizer = torch.optim.AdamW(correction.parameters(), lr=0.001, weight_decay=0.02)
        history = []
        for epoch in range(1, args.epochs + 1):
            correction.train()
            for indices in torch.randperm(len(targets), device=device).split(128):
                optimizer.zero_grad(set_to_none=True)
                output = train_baseline[indices] + correction(train_features[indices, :size])
                loss = nn.functional.cross_entropy(output, targets[indices], weight=weights)
                penalty = sum(p.square().mean() for p in correction.parameters())
                (loss + 0.03 * penalty).backward()
                optimizer.step()
            correction.eval()
            with torch.inference_mode():
                validation_correction = correction(
                    features["val"][:, :size].to(device)
                ).cpu().numpy()
            # Selection uses only validation people/videos. The test is stored
            # only once a winner is chosen; do not tune after reading it.
            for alpha in (0.25, 0.5, 1.0):
                selected = baseline["val"] + alpha * validation_correction
                scores = group_reports(selected, arrays["val"])
                ravdess = scores["ravdess_normal_early"]
                baseline_ravdess = baselines["val"]["ravdess_normal_early"]
                crema = scores["cremad_unknown_intensity"]
                baseline_crema = baselines["val"]["cremad_unknown_intensity"]
                score = 0.8 * ravdess["macro_f1_supported"] + 0.2 * crema["macro_f1_supported"]
                # Reject candidates that sacrifice anger or fear in validation.
                protects = all(
                    ravdess["per_class"][label]["recall"]
                    >= baseline_ravdess["per_class"][label]["recall"] - 0.05
                    for label in ("anger", "fear")
                )
                protects &= (
                    ravdess["macro_f1_supported"] > baseline_ravdess["macro_f1_supported"]
                    and crema["macro_f1_supported"]
                    >= baseline_crema["macro_f1_supported"] - 0.015
                    and sum(ravdess["per_class"][label]["recall"] for label in ("anger", "fear"))
                    > sum(baseline_ravdess["per_class"][label]["recall"] for label in ("anger", "fear"))
                )
                if protects and (best is None or score > best["selection_score"]):
                    best = {
                        "kind": kind, "epoch": epoch, "alpha": alpha,
                        "selection_score": score, "validation": scores,
                        "state": {k: v.detach().cpu().clone() for k, v in correction.state_dict().items()},
                    }
            history.append({"epoch": epoch, "best_score": best["selection_score"] if best else None})
        print("trained", kind, "best", best["selection_score"] if best else None, flush=True)
    if best is None:
        raise RuntimeError("No candidate passed the validation guard")
    size = 21 if best["kind"] == "logits" else raw["train"].shape[1]
    chosen = nn.Linear(size, len(CLASSES)).to(device)
    chosen.load_state_dict(best["state"])
    chosen.eval()
    with torch.inference_mode():
        correction_test = chosen(features["test"][:, :size].to(device)).cpu().numpy()
    selected_test = group_reports(baseline["test"] + best["alpha"] * correction_test, arrays["test"])
    payload = {
        "status": "experimental-not-deployed",
        "source_checkpoint": str(args.checkpoint),
        "split": "RAVDESS actors 01-16/17-20/21-24; CREMA-D actors 1001-1071/1072-1081/1082-1091",
        "selection": {k: v for k, v in best.items() if k != "state"},
        "baseline_validation": baselines["val"],
        "baseline_test": baselines["test"],
        "candidate_test": selected_test,
    }
    (args.output / "temporal-report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state": best["state"], "kind": best["kind"], "alpha": best["alpha"],
                "mean": mean, "std": std, "source_checkpoint": str(args.checkpoint)},
               args.output / "temporal-head.pt")
    print("selected", best["kind"], best["epoch"], best["alpha"], flush=True)
    for group in ("ravdess_normal_early", "cremad_low_medium", "cremad_unknown_intensity"):
        if group in selected_test:
            before = baselines["test"][group]
            after = selected_test[group]
            print(group, "N", before["examples"], "F1", before["macro_f1_supported"], "->", after["macro_f1_supported"],
                  "anger", before["per_class"]["anger"]["recall"], "->", after["per_class"]["anger"]["recall"],
                  "fear", before["per_class"]["fear"]["recall"], "->", after["per_class"]["fear"]["recall"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("training/runs/vit-small-seven-glasses-v1/best_target.pt"))
    parser.add_argument("--ravdess", type=Path, default=Path("data/ravdess/faces-seven"))
    parser.add_argument("--ravdess-test", type=Path, default=Path("data/ravdess/faces-seven-test-complete"))
    parser.add_argument("--cremad", type=Path, default=Path("data/cremad/faces-facevote-v1"))
    parser.add_argument("--summary", type=Path, default=Path(r"D:\CREMA-D-gitlab\summaryTable.csv"))
    parser.add_argument("--output", type=Path, default=Path("training/runs/temporal-v1"))
    parser.add_argument("--min-confidence", type=float, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
