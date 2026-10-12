from __future__ import annotations

import argparse
from pathlib import Path

import torch


SEVEN_CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove contempt from an eight-class ViT checkpoint")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    source_classes = tuple(checkpoint["classes"])
    if "contempt" not in source_classes or len(source_classes) != 8:
        raise ValueError(f"Expected an eight-class checkpoint containing contempt, found {source_classes}")
    keep = [source_classes.index(label) for label in SEVEN_CLASSES]
    state = checkpoint["model_state"]
    output_tensors = [
        key for key, value in state.items()
        if value.ndim in (1, 2) and value.shape[0] == len(source_classes) and key.rsplit(".", 1)[-1] in ("weight", "bias")
    ]
    if len(output_tensors) != 2:
        raise ValueError(f"Could not identify classifier tensors: {output_tensors}")
    for key in output_tensors:
        state[key] = state[key][keep].clone()
    checkpoint["classes"] = SEVEN_CLASSES
    checkpoint["class_counts"] = [checkpoint["class_counts"][index] for index in keep]
    checkpoint["class_weights"] = [checkpoint["class_weights"][index] for index in keep]
    checkpoint["converted_from"] = str(args.input)
    for key in ("validation", "validation_confusion", "target_validation"):
        checkpoint.pop(key, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved seven-class checkpoint: {args.output}")


if __name__ == "__main__":
    main()
