from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server


MAPPING = {"AN": "anger", "DI": "disgust", "AF": "fear", "HA": "happiness", "NE": "neutral", "SA": "sadness", "SU": "surprise"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark actual 512-pixel product preprocessing on unseen KDEF frontal photographs")
    parser.add_argument("--archive", type=Path, default=Path(r"D:\KDEF\KDEF_and_AKDEF.zip"))
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "kdef-product-benchmark.json")
    args = parser.parse_args()
    examples = []
    pattern = re.compile(r"[AB][FM](3[1-5])(AN|DI|AF|HA|NE|SA|SU)S")
    with ZipFile(args.archive) as archive:
        for entry in sorted(archive.namelist()):
            match = pattern.fullmatch(Path(entry).stem.upper())
            if not match:
                continue
            image = cv2.imdecode(np.frombuffer(archive.read(entry), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot decode {entry}")
            scale = 512 / max(image.shape[:2])
            image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            # Include the actual camera upload JPEG encoding/decoding step.
            success, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise RuntimeError(f"Cannot encode {entry}")
            image = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            examples.append((entry, image, MAPPING[match.group(2)]))
    reports = {}
    models = {"previous": ROOT / "models" / "vit_emotion_seven_pre_kdef_int8.onnx", "current": server.TRANSFORMER_MODEL_PATH}
    for name, model_path in models.items():
        server.CLASSIFIER_SPECS = (server.ClassifierSpec("transformer", "ViT-S/16 seven", "Pure ViT", model_path, 7),)
        recognizer = server.EmotionRecognizer()
        for _, image, _ in examples[:3]:
            recognizer.analyze(image)
        rows = []
        counts = Counter()
        for entry, image, label in examples:
            start = perf_counter()
            result = recognizer.analyze(image)
            duration = (perf_counter() - start) * 1000
            predicted = result["faces"][0]["emotion"]["id"] if result["faces"] else None
            counts[f"support/{label}"] += 1
            counts[f"correct/{label}"] += predicted == label
            rows.append({"file": entry, "label": label, "predicted": predicted, "latency_ms": duration})
        latencies = np.array([row["latency_ms"] for row in rows])
        reports[name] = {
            "model": str(model_path), "examples": len(rows),
            "accuracy": sum(row["predicted"] == row["label"] for row in rows) / len(rows),
            "face_detection_failures": sum(row["predicted"] is None for row in rows),
            "latency_ms_mean": float(latencies.mean()), "latency_ms_median": float(np.median(latencies)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "recall": {label: counts[f"correct/{label}"] / counts[f"support/{label}"] for label in MAPPING.values()},
            "records": rows,
        }
        del recognizer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "original unseen KDEF frontal photos, resize longest edge 512, JPEG quality 85, actual product detection/crop/classification; not live webcam accuracy",
              "models": reports}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in result.items() if key != "records"} for name, result in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
