from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained emotion ViT to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = tuple(checkpoint["classes"])
    image_size = int(checkpoint["image_size"])
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=len(classes),
        **checkpoint.get('model_kwargs', {}),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    with torch.inference_mode():
        torch_output = model(example).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"input": example.numpy()})[0]
    difference = np.abs(torch_output - onnx_output)
    report = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.output),
        "model": checkpoint["model_name"],
        "model_kwargs": checkpoint.get('model_kwargs', {}),
        "classes": classes,
        "grayscale": checkpoint.get("grayscale", True),
        "mean": checkpoint.get("mean", [0.485, 0.456, 0.406]),
        "std": checkpoint.get("std", [0.229, 0.224, 0.225]),
        "input_shape": ["batch_size", 3, image_size, image_size],
        "output_shape": ["batch_size", len(classes)],
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "same_top_class": bool(torch_output.argmax() == onnx_output.argmax()),
        "training": {
            "best_epoch": checkpoint.get("epoch"),
            "class_weighting": checkpoint.get("class_weighting"),
            "class_counts": checkpoint.get("class_counts"),
            "validation": checkpoint.get("validation"),
            "target_validation": checkpoint.get("target_validation"),
        },
    }
    if args.checkpoint.name == "best.pt":
        test_report_path = args.checkpoint.parent / "test_report.json"
        if test_report_path.is_file():
            report["test"] = json.loads(test_report_path.read_text(encoding="utf-8"))
    elif args.checkpoint.name == "best_target.pt":
        target_report_path = args.checkpoint.parent / "target_candidate_report.json"
        if target_report_path.is_file():
            target_report = json.loads(target_report_path.read_text(encoding="utf-8"))
            if target_report.get("epoch") == checkpoint.get("epoch"):
                report["test"] = target_report.get("test")
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["max_absolute_difference"] > 1e-3 or not report["same_top_class"]:
        raise RuntimeError("ONNX verification exceeded the allowed numerical difference")


if __name__ == "__main__":
    main()
