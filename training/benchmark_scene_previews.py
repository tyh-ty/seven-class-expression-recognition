"""Diagnose public video previews using webcam preprocessing, without training."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import EmotionRecognizer, TRANSFORMER_MODEL_PATH


def main() -> None:
    directory = ROOT / "output" / "real-scene-preview-20260927"
    recognizer = EmotionRecognizer()
    videos = []
    thumbnails = []
    for path in sorted(directory.glob("*/*.mp4")):
        label = {"angry": "anger"}.get(path.stem, path.stem)
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not count or fps <= 0:
            raise RuntimeError(f"Unreadable video: {path}")
        step = max(1, round(fps / 5))
        frames = []
        for frame_index in range(0, count, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, original = capture.read()
            if not ok:
                continue
            height, width = original.shape[:2]
            scale = min(1.0, 512 / max(width, height))
            resized = cv2.resize(original, (round(width * scale), round(height * scale)))
            ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            result = recognizer.analyze(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
            nearest = max(result["faces"], key=lambda f: f["bbox"]["width"] * f["bbox"]["height"], default=None)
            row = {"frame": frame_index, "seconds": round(frame_index / fps, 3),
                   "faces": len(result["faces"]), "latencyMs": result["latencyMs"]}
            if nearest:
                row.update({"predicted": nearest["emotion"], "bbox": nearest["bbox"],
                            "scores": nearest["scores"],
                            "displayable": nearest["emotion"]["score"] >=
                            (0.32 if nearest["emotion"]["id"] == "fear" else 0.4)})
            frames.append(row)
            if frame_index in (0, (count // (2 * step)) * step, ((count - 1) // step) * step):
                thumbnail = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                thumbnail.thumbnail((320, 190))
                tile = Image.new("RGB", (340, 240), "#f3f6f9")
                tile.paste(thumbnail, ((340 - thumbnail.width) // 2, 10))
                text = f"{path.parent.name}/{label}  {frame_index/fps:.1f}s"
                predicted = row.get("predicted", {})
                text += f"\nmodel: {predicted.get('id', 'no_face')} {predicted.get('score', 0):.2f}"
                ImageDraw.Draw(tile).text((10, 202), text, fill="#122033")
                thumbnails.append(tile)
        capture.release()
        predictions = Counter(row.get("predicted", {}).get("id", "no_face") for row in frames)
        record = {"source": path.parent.name, "file": path.name, "source_clip_label": label,
                  "seconds": round(count / fps, 3), "sampled_frames": len(frames),
                  "detected_frames": sum(row["faces"] > 0 for row in frames),
                  "predictions": dict(predictions), "frames": frames}
        videos.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "frames"}), flush=True)
    report = {
        "scope": "Eight hand-selected official public movie/TV demonstration clips; diagnostic only. "
                 "Clip labels are NOT independently reviewed per-frame facial ground truth. "
                 "Not a held-out benchmark, training set, webcam accuracy or microexpression evaluation.",
        "model": str(TRANSFORMER_MODEL_PATH),
        "model_sha256": hashlib.sha256(TRANSFORMER_MODEL_PATH.read_bytes()).hexdigest(),
        "preprocessing": "Longest edge <=512, JPEG quality 85, deployed YuNet and ViT; largest detected face; 5fps sampling",
        "sources": ["https://caer-dataset.github.io/", "https://dfew-dataset.github.io/"],
        "videos": videos,
    }
    (directory / "product-diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (3 * 340, ((len(thumbnails) + 2) // 3) * 240), "white")
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % 3) * 340, (index // 3) * 240))
    sheet.save(directory / "product-contact.jpg", quality=92)


if __name__ == "__main__":
    main()
