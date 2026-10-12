from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from emotion_preprocessing import EmotionEvaluationTransform
from dataset_contract import (
    CLASSES,
    EXTRA_SOURCE_PREFIXES,
    IMAGE_SUFFIXES,
    SCHEMA_VERSION,
    resolve_target_validation_path,
    sha256_file,
    validate_class_layout,
    write_manifest,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class: dict[str, dict[str, float | int]]


class SampleWeightedLabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        smoothing: float,
        extra_source_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", weights)
        self.register_buffer("extra_source_weights", extra_source_weights)
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits, dim=1)
        negative_log_likelihood = -log_probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        smooth_loss = -log_probabilities.mean(dim=1)
        losses = (
            (1.0 - self.smoothing) * negative_log_likelihood
            + self.smoothing * smooth_loss
        )
        sample_weights = self.weights[targets]
        if is_extra is not None and self.extra_source_weights is not None:
            sample_weights = torch.where(
                is_extra,
                self.extra_source_weights[targets],
                sample_weights,
            )
        return (losses * sample_weights).sum() / sample_weights.sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ViT emotion classifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path,
                        help="Optional audited primary validation folder; otherwise use DATA/val.")
    parser.add_argument("--require-disjoint-validation", action="store_true",
                        help="Fail before training if primary/target validation bytes occur in training.")
    parser.add_argument("--output", type=Path, default=Path("training/runs/vit-small"))
    parser.add_argument(
        "--model",
        default="vit_small_patch16_224.augreg_in21k_ft_in1k",
        help="Any timm image-classification model with a replaceable classifier head.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Start from a previously fine-tuned checkpoint instead of ImageNet weights.",
    )
    parser.add_argument(
        "--target-validation-data",
        type=Path,
        help="Optional partial-class folder used to select a real-world adaptation candidate.",
    )
    parser.add_argument(
        "--primary-f1-tolerance",
        type=float,
        default=0.003,
        help="Maximum FERPlus validation macro-F1 drop allowed for the target candidate.",
    )
    parser.add_argument(
        "--target-min-support",
        type=int,
        default=5,
        help="Minimum target-validation examples required for a class to affect selection.",
    )
    parser.add_argument('--primary-f1-floor', type=float,
                        help='Optional absolute primary-validation macro-F1 floor for accepting a target checkpoint, independent of a weak initialization.')
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse-sqrt", "effective"),
        default="inverse-sqrt",
    )
    parser.add_argument(
        "--extra-source-class-weighting",
        choices=("inherit", "none", "inverse-sqrt", "effective"),
        default="inherit",
        help=(
            "Class weighting for web_/ravdess_/cremad_/kdef_/ckplus_ training samples. 'inherit' uses the "
            "combined-dataset weights; 'none' prevents a rare FERPlus class from "
            "being over-weighted in a balanced adaptation source."
        ),
    )
    parser.add_argument("--effective-beta", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep the deployed grayscale pipeline by default; --no-grayscale trains RGB candidates.")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--train-input-projection", action="store_true",
                        help="Also update patch embedding after the freeze period when only late blocks are trainable, for a converted RGB initialization.")
    parser.add_argument(
        "--trainable-backbone-blocks",
        type=int,
        help="After the freeze period, update only the final N Transformer blocks and final norm.",
    )
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-scope", choices=("base", "base-and-correct-extra"), default="base",
                        help="Optionally also preserve the teacher on new-source training examples where its top prediction agrees with the source label. Never filters validation or changes labels.")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Optional separate seven-class teacher, including a different ViT architecture. Uses its recorded color preprocessing; defaults to the initial checkpoint.")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--no-evaluate-test",
        action="store_true",
        help="Do not read the test split during training; reserve it for a later blind evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic PyTorch settings where supported and record the actual setting.",
    )
    parser.add_argument(
        "--save-all-epochs",
        action="store_true",
        help="Keep an evaluation checkpoint for every epoch for later multi-domain selection.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic


def git_provenance(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def package_versions() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": getattr(__import__("torchvision"), "__version__", None),
        "timm": getattr(timm, "__version__", None),
    }
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        versions["onnxruntime"] = None
    if torch.cuda.is_available():
        versions["cuda"] = torch.version.cuda
        versions["gpu"] = torch.cuda.get_device_name(0)
    else:
        versions["cuda"] = None
        versions["gpu"] = None
    return versions


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int, grayscale: bool = True) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            *([transforms.Grayscale(num_output_channels=3)] if grayscale else []),
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = EmotionEvaluationTransform(image_size=image_size, grayscale=grayscale)
    return train_transform, evaluation_transform


class SourceAwareImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, bool]:
        image, target = super().__getitem__(index)
        filename = Path(self.samples[index][0]).name
        return image, target, filename.startswith(EXTRA_SOURCE_PREFIXES)


def load_split(
    path: Path,
    transform: transforms.Compose,
    allow_empty: bool = False,
    source_aware: bool = False,
) -> datasets.ImageFolder:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    validate_class_layout(path, allow_empty=allow_empty)
    dataset_type = SourceAwareImageFolder if source_aware else datasets.ImageFolder
    dataset = dataset_type(path, transform=transform, allow_empty=allow_empty)
    if tuple(dataset.classes) != CLASSES:
        raise ValueError(
            f"Expected class folders {CLASSES}, found {tuple(dataset.classes)} in {path}"
        )
    return dataset


def make_loader(
    dataset: datasets.ImageFolder,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def class_counts(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = torch.bincount(
        torch.tensor(dataset.targets, dtype=torch.long),
        minlength=len(CLASSES),
    ).float()
    if torch.any(counts == 0):
        missing = [CLASSES[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training split contains empty classes: {missing}")
    return counts


def source_class_counts(
    dataset: datasets.ImageFolder, prefixes: tuple[str, ...]
) -> torch.Tensor:
    counts = torch.zeros(len(CLASSES), dtype=torch.float32)
    for path, target in dataset.samples:
        if Path(path).name.startswith(prefixes):
            counts[target] += 1
    return counts


def class_weights(
    counts: torch.Tensor,
    method: str,
    effective_beta: float,
    device: torch.device,
) -> torch.Tensor:
    if method == "none":
        weights = torch.ones_like(counts)
    elif method == "inverse-sqrt":
        weights = counts.rsqrt()
    elif method == "effective":
        if not 0 < effective_beta < 1:
            raise ValueError("--effective-beta must be between zero and one")
        weights = (1.0 - effective_beta) / (1.0 - effective_beta**counts)
    else:
        raise ValueError(f"Unknown class weighting method: {method}")
    return (weights / weights.mean()).to(device)


def metrics_from_confusion(confusion: torch.Tensor, loss: float) -> EpochMetrics:
    matrix = confusion.float()
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = matrix.sum().clamp_min(1)
    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(CLASSES)
    }
    return EpochMetrics(
        loss=round(loss, 6),
        accuracy=round(float(true_positive.sum() / total), 6),
        macro_f1=round(float(f1.mean()), 6),
        balanced_accuracy=round(float(recall.mean()), 6),
        per_class=per_class,
    )


def supported_metrics_from_confusion(
    confusion: torch.Tensor, minimum_support: int = 1
) -> dict[str, float | list[str]]:
    matrix = confusion.float()
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    supported = support > 0
    selected = support >= minimum_support
    if not bool(selected.any()):
        selected = supported
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "macro_f1_supported": round(float(f1[supported].mean()), 6),
        "balanced_accuracy_supported": round(float(recall[supported].mean()), 6),
        "supported_classes": [
            label for index, label in enumerate(CLASSES) if bool(supported[index])
        ],
        "macro_f1_selected": round(float(f1[selected].mean()), 6),
        "balanced_accuracy_selected": round(float(recall[selected].mean()), 6),
        "selected_classes": [
            label for index, label in enumerate(CLASSES) if bool(selected[index])
        ],
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    max_batches: int | None = None,
    amp_dtype: torch.dtype = torch.float16,
    teacher_model: nn.Module | None = None,
    distillation_weight: float = 0.0,
    distillation_temperature: float = 2.0,
    distillation_scope: str = "base",
) -> tuple[EpochMetrics, torch.Tensor, int]:
    if distillation_scope not in ("base", "base-and-correct-extra"):
        raise ValueError(f"Unknown distillation scope: {distillation_scope}")
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    total_loss = 0.0
    total_examples = 0
    optimizer_steps = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    planned_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if len(batch) == 3:
            images, targets, is_extra = batch
        else:
            images, targets = batch
            is_extra = None
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_extra is not None:
            is_extra = is_extra.to(device, non_blocking=True)
        amp_enabled = device.type == "cuda"
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets, is_extra)
                if (
                    training
                    and teacher_model is not None
                    and is_extra is not None
                    and distillation_weight > 0
                ):
                    selected = ~is_extra
                    if distillation_scope == "base-and-correct-extra":
                        with torch.no_grad():
                            all_teacher_logits = teacher_model(images)
                        selected = selected | (all_teacher_logits.argmax(dim=1) == targets)
                        teacher_logits = all_teacher_logits[selected]
                    elif bool(selected.any()):
                        with torch.no_grad():
                            teacher_logits = teacher_model(images[selected])
                    if bool(selected.any()):
                        temperature = distillation_temperature
                        distillation_loss = F.kl_div(
                            F.log_softmax(logits[selected] / temperature, dim=1),
                            F.softmax(teacher_logits / temperature, dim=1),
                            reduction="batchmean",
                        ) * (temperature**2)
                        loss = loss + distillation_weight * distillation_loss
                scaled_loss = loss / accumulation_steps
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(scaled_loss).backward()
                should_step = (batch_index + 1) % accumulation_steps == 0
                is_last = batch_index + 1 == planned_batches
                if should_step or is_last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        optimizer_steps += 1
                    optimizer.zero_grad(set_to_none=True)

        predictions = logits.argmax(dim=1)
        batch_confusion = torch.bincount(
            targets.detach().cpu() * len(CLASSES) + predictions.detach().cpu(),
            minlength=len(CLASSES) ** 2,
        ).reshape(len(CLASSES), len(CLASSES))
        confusion += batch_confusion
        total_loss += float(loss.detach()) * targets.size(0)
        total_examples += targets.size(0)

    if total_examples == 0:
        raise RuntimeError("No batches were processed")
    return (
        metrics_from_confusion(confusion, total_loss / total_examples),
        confusion,
        optimizer_steps,
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DistillationTeacher(nn.Module):
    """Adapt student-normalized RGB tensors to the teacher's recorded input contract."""
    def __init__(self, model: nn.Module, grayscale: bool,
                 mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        self.model = model
        self.grayscale = grayscale
        self.register_buffer('student_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('student_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer('teacher_mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('teacher_std', torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer('luminance', torch.tensor((.299, .587, .114)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Keep preprocessing float32 even when the network runs under autocast.
        rgb = images.float() * self.student_std + self.student_mean
        if self.grayscale:
            rgb = (rgb * self.luminance).sum(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        return self.model((rgb - self.teacher_mean) / self.teacher_std)


def main() -> None:
    args = parse_args()
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be positive")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs cannot be negative")
    if args.trainable_backbone_blocks is not None and args.trainable_backbone_blocks < 1:
        raise ValueError("--trainable-backbone-blocks must be positive")
    if args.primary_f1_tolerance < 0:
        raise ValueError("--primary-f1-tolerance cannot be negative")
    if args.primary_f1_floor is not None and not 0 <= args.primary_f1_floor <= 1:
        raise ValueError('--primary-f1-floor must lie within[0,1]')
    if args.target_min_support < 1:
        raise ValueError("--target-min-support must be positive")
    if args.distillation_weight < 0:
        raise ValueError("--distillation-weight cannot be negative")
    if args.distillation_temperature <= 0:
        raise ValueError("--distillation-temperature must be positive")
    if args.teacher_checkpoint and args.distillation_weight <= 0:
        raise ValueError("--teacher-checkpoint requires a positive --distillation-weight")
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    train_transform, evaluation_transform = build_transforms(args.image_size, grayscale=args.grayscale)
    train_dataset = load_split(
        args.data / "train", train_transform, source_aware=True
    )
    validation_path = args.validation_data or args.data / "val"
    validation_dataset = load_split(validation_path, evaluation_transform)
    test_path = args.data / "test"
    test_dataset = (
        None
        if args.no_evaluate_test
        else load_split(test_path, evaluation_transform) if test_path.is_dir() else None
    )
    resolved_target_path: Path | None = None
    target_validation_dataset = None
    if args.target_validation_data:
        resolved_target_path, _ = resolve_target_validation_path(args.target_validation_data)
        target_validation_dataset = load_split(
            resolved_target_path, evaluation_transform, allow_empty=True
        )

    manifest_roots: dict[str, Path] = {
        "train": args.data / "train",
        "validation": validation_path,
    }
    if test_path.is_dir():
        manifest_roots["test"] = test_path
    if resolved_target_path is not None:
        manifest_roots["target_validation"] = resolved_target_path
    manifest_info = write_manifest(args.output / "dataset-manifest.jsonl", manifest_roots)
    if args.require_disjoint_validation:
        from collections import defaultdict
        by_hash = defaultdict(list)
        with (args.output / "dataset-manifest.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                by_hash[row["sha256"]].append(row)
        overlaps = [rows for rows in by_hash.values()
                    if any(row["role"] == "train" for row in rows)
                    and any(row["role"] in {"validation", "target_validation"} for row in rows)]
        audit = {"scope": "Exact file bytes only; this does not exclude all identity or perceptual overlaps.",
                 "cross_train_validation_hash_groups": len(overlaps), "overlaps": overlaps}
        save_json(args.output / "validation-overlap-audit.json", audit)
        if overlaps:
            raise ValueError("Training/validation exact-content overlap; inspect validation-overlap-audit.json before training")
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    initial_checkpoint_info = None
    if args.initial_checkpoint:
        initial_checkpoint_info = {
            "path": str(args.initial_checkpoint.resolve()),
            "size_bytes": args.initial_checkpoint.stat().st_size,
            "sha256": sha256_file(args.initial_checkpoint),
        }
    teacher_checkpoint_info = None
    if args.distillation_weight > 0:
        teacher_path = args.teacher_checkpoint or args.initial_checkpoint
        if teacher_path is None:
            raise ValueError("Distillation requires --teacher-checkpoint or --initial-checkpoint")
        teacher_checkpoint_info = {"path": str(teacher_path.resolve()),
                                   "size_bytes": teacher_path.stat().st_size,
                                   "sha256": sha256_file(teacher_path)}
    root = Path(__file__).resolve().parents[1]
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "argv": sys.argv,
        "args": serializable_args,
        "paths": {
            "data": str(args.data.resolve()),
            "target_validation": str(resolved_target_path.resolve())
            if resolved_target_path is not None
            else None,
            "output": str(args.output.resolve()),
        },
        "classes": CLASSES,
        "image_size": args.image_size,
        "grayscale": args.grayscale,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "extra_source_prefixes": EXTRA_SOURCE_PREFIXES,
        "dataset_manifest": manifest_info,
        "initial_checkpoint": initial_checkpoint_info,
        "teacher_checkpoint": teacher_checkpoint_info,
        "environment": package_versions(),
        "git": git_provenance(root),
        "device": str(device),
        "deterministic": args.deterministic,
    }
    run_config_path = args.output / "run_config.json"
    save_json(run_config_path, run_config)
    run_config_hash = sha256_file(run_config_path)

    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, device
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.workers, False, device
    )
    test_loader = (
        make_loader(test_dataset, args.batch_size, args.workers, False, device)
        if test_dataset is not None
        else None
    )
    target_validation_loader = (
        make_loader(
            target_validation_dataset, args.batch_size, args.workers, False, device
        )
        if target_validation_dataset is not None
        else None
    )

    counts = class_counts(train_dataset)
    weights = class_weights(
        counts,
        args.class_weighting,
        args.effective_beta,
        device,
    )
    extra_counts = source_class_counts(
        train_dataset, EXTRA_SOURCE_PREFIXES
    )
    if args.extra_source_class_weighting == "inherit":
        extra_weights = None
    else:
        if torch.any(extra_counts == 0):
            missing = [
                CLASSES[index]
                for index, count in enumerate(extra_counts)
                if count == 0
            ]
            raise ValueError(f"Extra training source contains empty classes: {missing}")
        extra_weights = class_weights(
            extra_counts,
            args.extra_source_class_weighting,
            args.effective_beta,
            device,
        )

    initial_checkpoint = None
    model_name = args.model
    if args.initial_checkpoint:
        initial_checkpoint = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if tuple(initial_checkpoint["classes"]) != CLASSES:
            raise ValueError("Initial checkpoint classes do not match this training run")
        if int(initial_checkpoint["image_size"]) != args.image_size:
            raise ValueError("Initial checkpoint image size does not match --image-size")
        model_name = initial_checkpoint["model_name"]

    model_kwargs = initial_checkpoint.get('model_kwargs', {}) if initial_checkpoint is not None else {}
    model = timm.create_model(
        model_name,
        pretrained=args.pretrained and initial_checkpoint is None,
        num_classes=len(CLASSES),
        drop_rate=0.1,
        **model_kwargs,
    ).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state"])
    teacher_model = None
    if args.distillation_weight > 0:
        teacher_checkpoint = (torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
                              if args.teacher_checkpoint else initial_checkpoint)
        if tuple(teacher_checkpoint["classes"]) != CLASSES:
            raise ValueError("Teacher class order differs from the seven-class contract")
        if int(teacher_checkpoint["image_size"]) != args.image_size:
            raise ValueError("Teacher image size differs from this run")
        if args.grayscale and not teacher_checkpoint.get("grayscale", True):
            raise ValueError("Cannot supply a color teacher from grayscale student inputs")
        teacher_network = timm.create_model(
            teacher_checkpoint["model_name"],
            pretrained=False,
            num_classes=len(CLASSES),
            drop_rate=0.1,
            **teacher_checkpoint.get('model_kwargs', {}),
        ).to(device)
        teacher_network.load_state_dict(teacher_checkpoint["model_state"])
        # Preserve the historical same-checkpoint input path exactly. Only an
        # explicit external teacher enables normalization/color adaptation.
        teacher_model = (DistillationTeacher(teacher_network,
            teacher_checkpoint.get("grayscale", True),
            tuple(teacher_checkpoint.get("mean", IMAGENET_MEAN)),
            tuple(teacher_checkpoint.get("std", IMAGENET_STD))).to(device)
            if args.teacher_checkpoint else teacher_network)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    criterion = SampleWeightedLabelSmoothingLoss(
        weights=weights,
        smoothing=args.label_smoothing,
        extra_source_weights=extra_weights,
    )
    head_parameters = list(model.get_classifier().parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) not in head_parameter_ids
    ]
    backbone_parameters = [parameter for _, parameter in backbone_named_parameters]
    block_count = len(model.blocks) if hasattr(model, "blocks") else None
    if args.trainable_backbone_blocks is not None:
        if block_count is None:
            raise ValueError("--trainable-backbone-blocks requires a model with Transformer blocks")
        if args.trainable_backbone_blocks > block_count:
            raise ValueError(
                f"--trainable-backbone-blocks cannot exceed the model's {block_count} blocks"
            )
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": args.learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and not use_bfloat16,
        init_scale=4096.0,
    )
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_path = args.output / "best.pt"
    best_target_path = args.output / "best_target.pt"
    initial_primary_macro_f1: float | None = None
    best_target_macro_f1 = -1.0

    def checkpoint_payload(
        epoch: int,
        validation_metrics: EpochMetrics,
        validation_confusion: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "model_state": model.state_dict(),
            "classes": CLASSES,
            "image_size": args.image_size,
            "grayscale": args.grayscale,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "class_counts": counts.int().tolist(),
            "class_weights": weights.cpu().tolist(),
            "class_weighting": args.class_weighting,
            "extra_source_class_counts": extra_counts.int().tolist(),
            "extra_source_class_weights": (
                extra_weights.cpu().tolist() if extra_weights is not None else None
            ),
            "extra_source_class_weighting": args.extra_source_class_weighting,
            "label_smoothing": args.label_smoothing,
            "backbone_learning_rate": args.learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "initial_checkpoint": (
                str(args.initial_checkpoint) if args.initial_checkpoint else None
            ),
            "initial_checkpoint_sha256": (
                initial_checkpoint_info["sha256"] if initial_checkpoint_info else None
            ),
            "run_config": str(run_config_path),
            "run_config_sha256": run_config_hash,
            "dataset_manifest": manifest_info,
            "extra_source_prefixes": EXTRA_SOURCE_PREFIXES,
            "deterministic": args.deterministic,
            "distillation_weight": args.distillation_weight,
            "distillation_temperature": args.distillation_temperature,
            "distillation_scope": args.distillation_scope,
            "teacher_checkpoint": teacher_checkpoint_info,
            "trainable_backbone_blocks": args.trainable_backbone_blocks,
            "train_input_projection": args.train_input_projection,
            "epoch": epoch,
            "validation": asdict(validation_metrics),
            "validation_confusion": validation_confusion.tolist(),
        }

    print(
        f"device={device}, model={model_name}, "
        f"pretrained={args.pretrained and initial_checkpoint is None}, "
        f"train={len(train_dataset)}, val={len(validation_dataset)}, "
        f"test={len(test_dataset) if test_dataset else 0}",
        flush=True,
    )
    if target_validation_dataset is not None:
        print(
            f"target_validation={len(target_validation_dataset)}, "
            f"primary_f1_tolerance={args.primary_f1_tolerance}, "
            f"target_min_support={args.target_min_support}",
            flush=True,
        )
    print(
        f"class_weighting={args.class_weighting}, "
        f"counts={counts.int().tolist()}, weights={weights.cpu().tolist()}",
        flush=True,
    )
    print(
        f"extra_source_class_weighting={args.extra_source_class_weighting}, "
        f"counts={extra_counts.int().tolist()}, "
        f"weights={extra_weights.cpu().tolist() if extra_weights is not None else 'inherit'}",
        flush=True,
    )
    print(
        f"backbone_lr={args.learning_rate}, head_lr={args.head_learning_rate}",
        flush=True,
    )
    if teacher_model is not None:
        print(
            f"distillation_weight={args.distillation_weight}, "
            f"distillation_temperature={args.distillation_temperature}",
            flush=True,
        )
    if args.initial_checkpoint:
        print(f"initial_checkpoint={args.initial_checkpoint}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        print(f"amp_dtype={amp_dtype}", flush=True)

    if initial_checkpoint is not None:
        initial_metrics, initial_confusion, _ = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
            amp_dtype=amp_dtype,
        )
        best_macro_f1 = initial_metrics.macro_f1
        initial_primary_macro_f1 = initial_metrics.macro_f1
        baseline_checkpoint = checkpoint_payload(0, initial_metrics, initial_confusion)
        temporary_path = best_path.with_suffix(".tmp")
        torch.save(baseline_checkpoint, temporary_path)
        temporary_path.replace(best_path)
        save_json(args.output / "initial_validation.json", asdict(initial_metrics))
        save_json(args.output / "best_validation.json", asdict(initial_metrics))
        print(
            f"initial_val_loss={initial_metrics.loss:.4f} "
            f"initial_val_f1={initial_metrics.macro_f1:.4f}",
            flush=True,
        )
        if target_validation_loader is not None:
            initial_target_metrics, initial_target_confusion, _ = run_epoch(
                model,
                target_validation_loader,
                criterion,
                device,
                max_batches=args.max_eval_batches,
                amp_dtype=amp_dtype,
            )
            initial_target_supported = supported_metrics_from_confusion(
                initial_target_confusion, args.target_min_support
            )
            best_target_macro_f1 = float(
                initial_target_supported["macro_f1_selected"]
            )
            temporary_target_path = best_target_path.with_suffix(".tmp")
            torch.save(baseline_checkpoint, temporary_target_path)
            temporary_target_path.replace(best_target_path)
            save_json(
                args.output / "initial_target_validation.json",
                {
                    "metrics": asdict(initial_target_metrics),
                    "supported_metrics": initial_target_supported,
                    "confusion": initial_target_confusion.tolist(),
                },
            )
            print(
                f"initial_target_f1={best_target_macro_f1:.4f}",
                flush=True,
            )

    for epoch in range(1, args.epochs + 1):
        backbone_frozen = epoch <= args.freeze_backbone_epochs
        first_trainable_block = (
            block_count - args.trainable_backbone_blocks
            if block_count is not None and args.trainable_backbone_blocks is not None
            else None
        )
        for name, parameter in backbone_named_parameters:
            if backbone_frozen:
                trainable = False
            elif first_trainable_block is None:
                trainable = True
            else:
                trainable = name.startswith("norm.") or any(
                    name.startswith(f"blocks.{index}.")
                    for index in range(first_trainable_block, block_count)
                ) or (args.train_input_projection and name.startswith("patch_embed.proj."))
            parameter.requires_grad_(trainable)
        started_at = perf_counter()
        train_metrics, _, optimizer_steps = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            max_batches=args.max_train_batches,
            amp_dtype=amp_dtype,
            teacher_model=teacher_model,
            distillation_weight=args.distillation_weight,
            distillation_temperature=args.distillation_temperature,
            distillation_scope=args.distillation_scope,
        )
        validation_metrics, validation_confusion, _ = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
            amp_dtype=amp_dtype,
        )
        target_metrics = None
        target_confusion = None
        target_supported = None
        if target_validation_loader is not None:
            target_metrics, target_confusion, _ = run_epoch(
                model,
                target_validation_loader,
                criterion,
                device,
                max_batches=args.max_eval_batches,
                amp_dtype=amp_dtype,
            )
            target_supported = supported_metrics_from_confusion(
                target_confusion, args.target_min_support
            )
        if optimizer_steps:
            scheduler.step()
        else:
            print("warning: optimizer made no update; learning-rate scheduler was not advanced")
        record = {
            "epoch": epoch,
            "backbone_frozen": backbone_frozen,
            "trainable_backbone_blocks": (
                0 if backbone_frozen else args.trainable_backbone_blocks
            ),
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            "duration_seconds": round(perf_counter() - started_at, 2),
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
        }
        if target_metrics is not None and target_supported is not None:
            record["target_validation"] = {
                "metrics": asdict(target_metrics),
                "supported_metrics": target_supported,
            }
        history.append(record)
        save_json(args.output / "history.json", history)
        target_text = (
            f" target_f1={float(target_supported['macro_f1_selected']):.4f}"
            if target_supported is not None
            else ""
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.loss:.4f} "
            f"train_f1={train_metrics.macro_f1:.4f} "
            f"val_loss={validation_metrics.loss:.4f} "
            f"val_f1={validation_metrics.macro_f1:.4f}{target_text}",
            flush=True,
        )

        if args.save_all_epochs:
            epoch_checkpoint = checkpoint_payload(
                epoch, validation_metrics, validation_confusion
            )
            if (
                target_metrics is not None
                and target_supported is not None
                and target_confusion is not None
            ):
                epoch_checkpoint["target_validation"] = {
                    "metrics": asdict(target_metrics),
                    "supported_metrics": target_supported,
                    "confusion": target_confusion.tolist(),
                }
            torch.save(epoch_checkpoint, args.output / f"epoch-{epoch:03d}.pt")

        if (
            target_supported is not None
            and target_confusion is not None
            and initial_primary_macro_f1 is not None
            and validation_metrics.macro_f1
            >= initial_primary_macro_f1 - args.primary_f1_tolerance
            and (args.primary_f1_floor is None or validation_metrics.macro_f1 >= args.primary_f1_floor)
            and float(target_supported["macro_f1_selected"]) > best_target_macro_f1
        ):
            best_target_macro_f1 = float(target_supported["macro_f1_selected"])
            target_checkpoint = checkpoint_payload(
                epoch, validation_metrics, validation_confusion
            )
            target_checkpoint["target_validation"] = {
                "metrics": asdict(target_metrics),
                "supported_metrics": target_supported,
                "confusion": target_confusion.tolist(),
            }
            temporary_target_path = best_target_path.with_suffix(".tmp")
            torch.save(target_checkpoint, temporary_target_path)
            temporary_target_path.replace(best_target_path)
            save_json(
                args.output / "best_target_validation.json",
                target_checkpoint["target_validation"],
            )

        if validation_metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = validation_metrics.macro_f1
            epochs_without_improvement = 0
            checkpoint = checkpoint_payload(
                epoch, validation_metrics, validation_confusion
            )
            temporary_path = best_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(best_path)
            save_json(args.output / "best_validation.json", checkpoint["validation"])
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"early_stop epoch={epoch} best_macro_f1={best_macro_f1:.4f}",
                    flush=True,
                )
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if test_loader is not None:
        test_metrics, test_confusion, _ = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
            amp_dtype=amp_dtype,
        )
        save_json(
            args.output / "test_report.json",
            {"metrics": asdict(test_metrics), "confusion": test_confusion.tolist()},
        )
        print(
            f"test_accuracy={test_metrics.accuracy:.4f} "
            f"test_macro_f1={test_metrics.macro_f1:.4f}",
            flush=True,
        )

    if target_validation_loader is not None and best_target_path.is_file():
        target_checkpoint = torch.load(
            best_target_path, map_location=device, weights_only=False
        )
        model.load_state_dict(target_checkpoint["model_state"])
        candidate_target_metrics, candidate_target_confusion, _ = run_epoch(
            model,
            target_validation_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
            amp_dtype=amp_dtype,
        )
        candidate_report: dict[str, Any] = {
            "epoch": target_checkpoint["epoch"],
            "primary_validation": target_checkpoint["validation"],
            "absolute_primary_f1_floor": args.primary_f1_floor,
            "meets_absolute_primary_f1_floor": (args.primary_f1_floor is None or
                target_checkpoint['validation']['macro_f1'] >= args.primary_f1_floor),
            "target_validation": {
                "metrics": asdict(candidate_target_metrics),
                "supported_metrics": supported_metrics_from_confusion(
                    candidate_target_confusion, args.target_min_support
                ),
                "confusion": candidate_target_confusion.tolist(),
            },
        }
        if test_loader is not None:
            candidate_test_metrics, candidate_test_confusion, _ = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                max_batches=args.max_eval_batches,
                amp_dtype=amp_dtype,
            )
            candidate_report["test"] = {
                "metrics": asdict(candidate_test_metrics),
                "confusion": candidate_test_confusion.tolist(),
            }
        save_json(args.output / "target_candidate_report.json", candidate_report)
        print(
            f"target_candidate_epoch={target_checkpoint['epoch']} "
            f"target_candidate_f1="
            f"{candidate_report['target_validation']['supported_metrics']['macro_f1_selected']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
