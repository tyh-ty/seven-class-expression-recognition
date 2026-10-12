from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose compatible checkpoints by parameter-name prefixes"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between zero and one")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    adapted = torch.load(args.adapted, map_location="cpu", weights_only=False)
    for key in ("model_name", "classes", "image_size"):
        if base[key] != adapted[key]:
            raise ValueError(f"Checkpoint field does not match: {key}")
    state = {key: value.clone() for key, value in base["model_state"].items()}
    selected = []
    for key, base_value in state.items():
        if not key.startswith(tuple(args.prefix)):
            continue
        adapted_value = adapted["model_state"][key]
        state[key] = (
            torch.lerp(base_value, adapted_value, args.alpha)
            if torch.is_floating_point(base_value)
            else adapted_value.clone()
        )
        selected.append(key)
    if not selected:
        raise ValueError("No parameters matched the requested prefixes")
    payload = dict(base)
    payload["model_state"] = state
    payload["composition"] = {
        "base": str(args.base),
        "adapted": str(args.adapted),
        "prefixes": args.prefix,
        "adapted_alpha": args.alpha,
        "parameters": selected,
    }
    payload["epoch"] = f"composed-{args.alpha:g}"
    payload["validation"] = None
    payload.pop("validation_confusion", None)
    payload.pop("target_validation", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {args.output}: {', '.join(selected)}")


if __name__ == "__main__":
    main()
