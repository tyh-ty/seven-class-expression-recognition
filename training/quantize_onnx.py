from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamically quantize ViT MatMul/Gemm weights")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"Quantized model: {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
