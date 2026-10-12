"""Create a small visually screened CAER adaptation experiment.

One already-inspected frame per selected clip. All other candidate frames stay
excluded. The source scene label is retained, never relabeled from model output.
This single assistant visual review is not independent expert ground truth.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
SELECTED = {
    "anger": [2, 3, 8, 14, 15, 18, 19, 29, 30, 43, 46, 54, 57, 62, 231, 233, 235, 236],
    "disgust": [66, 67, 68, 72, 74, 75, 78, 242, 244],
    "fear": [101, 102, 108, 112, 115, 118, 149, 250, 254, 255, 261],
    "happiness": [154, 155, 156, 157, 158, 159, 160, 163, 164, 165, 166, 169, 170, 171, 173,
                  262, 263, 265, 266, 267],
    "neutral": [174, 179, 183, 184, 185, 186, 188, 189, 269, 270, 271, 272],
    "sadness": [193, 196, 197, 198, 201, 203, 204, 205, 206, 208, 211, 212, 273, 277, 279, 280],
    "surprise": [215, 217, 219, 220, 222, 224, 282, 286, 288],
}
EYEWEAR = {15, 158, 165, 173, 183, 188, 265, 270}


def main() -> None:
    index_path = ROOT / "output/caer-pilot-review-20260927/eyewear-review-index.json"
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    review = ROOT / "data/caer/pilot-review-v1"
    base = ROOT / "data/seven-class/glasses-adaptation-v1"
    output = ROOT / "data/seven-class/caer-reviewed-pilot-v1"
    if output.exists():
        raise FileExistsError(output)
    decisions = []
    counts = Counter()
    for split in ("train", "val", "scene-val", "eyewear-val"):
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    # Replay all old training images, including the old glasses augmentation.
    # FERPlus validation is unchanged. No old or new test images enter this run.
    for split in ("train", "val"):
        for label in CLASSES:
            for source in sorted((base / split / label).glob("*.jpg")):
                os.link(source, output / split / label / source.name)
                counts[f"base/{split}/{label}"] += 1
    train_hashes, val_hashes = set(), set()
    for label, indices in SELECTED.items():
        for index in indices:
            row = rows[index]
            source = review / row["crop"]
            source_split, source_label, _ = Path(row["crop"]).parts
            if source_label != label or source_split not in ("train", "val"):
                raise ValueError(f"Review index changed: {index}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (train_hashes if source_split == "train" else val_hashes).add(digest)
            # A small repeat rate makes the new examples visible during replay.
            repeats = 8 if source_split == "train" else 1
            destination_split = "train" if source_split == "train" else "scene-val"
            for repeat in range(repeats):
                name = f"web_caer_reviewed_{digest[:20]}_r{repeat}.jpg"
                os.link(source, output / destination_split / label / name)
            if source_split == "val" and index in EYEWEAR:
                os.link(source, output / "eyewear-val" / label / source.name)
            counts[f"new_unique/{source_split}/{label}"] += 1
            decisions.append({**row, "index": index, "label": label, "split": source_split,
                              "sha256": digest, "eyewear_visible": index in EYEWEAR,
                              "decision": "retain source label for small pilot after visual review",
                              "training_repeats": repeats})
    if train_hashes & val_hashes:
        raise ValueError("Identical reviewed image in train and validation")
    report = {
        "review_method": "Single assistant visual inspection of contact sheets; not an independent human annotation study.",
        "review_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "selection": "One reviewed representative frame per clip. Keep only the listed frames; all others excluded.",
        "training_sources": "Existing seven-class + synthetic-glasses data replay; small real CAER facial subset.",
        "limits": ["Movie/TV footage is not webcam data.", "Subtle expressions are ambiguous; residual label noise remains.",
                   "Original clip splits retained, but actors/shows can overlap.",
                   "Only six new training eyewear frames and two validation eyewear frames; no evidence for real-eyewear fear generalization.",
                   "New scene validation has very few independent frames and is diagnostic only."],
        "counts": dict(counts), "decisions": decisions,
    }
    (output / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
