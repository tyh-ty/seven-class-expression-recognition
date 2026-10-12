from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linearly interpolate two compatible fine-tuned checkpoints"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between zero and one")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    adapted = torch.load(args.adapted, map_location="cpu", weights_only=False)
    for key in ("model_name", "classes", "image_size"):
        if base[key] != adapted[key]:
            raise ValueError(f"Checkpoint field does not match: {key}")
    if base.get('grayscale', True) != adapted.get('grayscale', True):
        raise ValueError('Checkpoint color preprocessing differs')
    for key, default in (('mean', (.485, .456, .406)), ('std', (.229, .224, .225))):
        if tuple(base.get(key, default)) != tuple(adapted.get(key, default)):
            raise ValueError(f'Checkpoint normalization differs: {key}')
    if base["model_state"].keys() != adapted["model_state"].keys():
        raise ValueError("Checkpoint model state keys do not match")

    state: dict[str, torch.Tensor] = {}
    for key, base_value in base["model_state"].items():
        adapted_value = adapted["model_state"][key]
        if base_value.shape != adapted_value.shape or base_value.dtype != adapted_value.dtype:
            raise ValueError(f'Checkpoint tensor differs in shape/dtype: {key}')
        if torch.is_floating_point(base_value):
            state[key] = torch.lerp(base_value, adapted_value, args.alpha)
        else:
            state[key] = adapted_value.clone()
    payload = dict(adapted)
    payload["model_state"] = state
    payload["interpolation"] = {
        "base": str(args.base),
        "adapted": str(args.adapted),
        "adapted_alpha": args.alpha,
        "base_sha256": sha256(args.base),
        "adapted_sha256": sha256(args.adapted),
    }
    payload["epoch"] = f"interpolated-{args.alpha:g}"
    # Parent scores do not describe the interpolated model.
    payload["validation"] = None
    payload.pop("validation_confusion", None)
    payload.pop("target_validation", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {args.output} with adapted alpha {args.alpha:g}")


if __name__ == "__main__":
    main()
