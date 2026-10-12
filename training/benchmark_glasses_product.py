from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server


MAPPING = {
    "AN": "anger",
    "DI": "disgust",
    "AF": "fear",
    "HA": "happiness",
    "NE": "neutral",
    "SA": "sadness",
    "SU": "surprise",
}


def detector(threshold: float) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        str(server.EmotionRecognizer._opencv_safe_path(server.FACE_DETECTOR_PATH)),
        "",
        (320, 320),
        threshold,
        0.3,
        5000,
    )


def primary_face(face_detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)
    if faces is None:
        return None
    return max(faces, key=lambda face: float(face[2] * face[3]))


def add_glasses(image: np.ndarray, landmarks: np.ndarray, seed: int) -> np.ndarray:
    result = image.copy()
    first_eye, second_eye = sorted(landmarks[:2], key=lambda point: point[0])
    eye_distance = float(np.linalg.norm(second_eye - first_eye))
    angle = float(np.degrees(np.arctan2(
        second_eye[1] - first_eye[1], second_eye[0] - first_eye[0]
    )))
    lens_axes = (
        max(8, int(round(eye_distance * 0.47))),
        max(6, int(round(eye_distance * (0.28 + 0.025 * (seed % 3))))),
    )
    thickness = max(2, int(round(eye_distance * 0.055)))
    centers = [tuple(np.rint(point).astype(int)) for point in (first_eye, second_eye)]

    tint = np.zeros_like(result)
    for center in centers:
        cv2.ellipse(tint, center, lens_axes, angle, 0, 360, (42, 42, 42), -1, cv2.LINE_AA)
    mask = np.any(tint != 0, axis=2)
    blended = cv2.addWeighted(result, 0.82, tint, 0.18, 0)
    result[mask] = blended[mask]

    frame_color = (18, 18, 18)
    for center in centers:
        cv2.ellipse(result, center, lens_axes, angle, 0, 360, frame_color, thickness, cv2.LINE_AA)
    bridge_start = (centers[0][0] + lens_axes[0], centers[0][1])
    bridge_end = (centers[1][0] - lens_axes[0], centers[1][1])
    cv2.line(result, bridge_start, bridge_end, frame_color, thickness, cv2.LINE_AA)

    direction = np.asarray((np.cos(np.radians(angle)), np.sin(np.radians(angle))))
    outer_left = np.asarray(centers[0]) - direction * lens_axes[0]
    outer_right = np.asarray(centers[1]) + direction * lens_axes[0]
    cv2.line(
        result,
        tuple(np.rint(outer_left).astype(int)),
        tuple(np.rint(outer_left + np.asarray((-eye_distance * 0.24, eye_distance * 0.05))).astype(int)),
        frame_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        result,
        tuple(np.rint(outer_right).astype(int)),
        tuple(np.rint(outer_right + np.asarray((eye_distance * 0.24, eye_distance * 0.05))).astype(int)),
        frame_color,
        thickness,
        cv2.LINE_AA,
    )

    glare = np.zeros_like(result)
    glare_mask = np.zeros(result.shape[:2], dtype=np.uint8)
    for index, center in enumerate(centers):
        cv2.ellipse(glare_mask, center, lens_axes, angle, 0, 360, 255, -1, cv2.LINE_AA)
        offset = int(round(eye_distance * (0.13 + 0.025 * ((seed + index) % 3))))
        start = (center[0] - offset, center[1] - lens_axes[1])
        end = (center[0] + offset, center[1] + lens_axes[1])
        cv2.line(glare, start, end, (235, 235, 235), max(2, thickness - 1), cv2.LINE_AA)
    glare = cv2.bitwise_and(glare, glare, mask=glare_mask)
    glare_pixels = np.any(glare != 0, axis=2)
    glare_blend = cv2.addWeighted(result, 0.68, glare, 0.32, 0)
    result[glare_pixels] = glare_blend[glare_pixels]
    return result


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def load_examples(archive_path: Path) -> list[dict[str, object]]:
    pattern = re.compile(r"[AB][FM](3[1-5])(AN|DI|AF|HA|NE|SA|SU)S")
    reference_detector = detector(0.35)
    examples: list[dict[str, object]] = []
    with ZipFile(archive_path) as archive:
        for entry in sorted(archive.namelist()):
            match = pattern.fullmatch(Path(entry).stem.upper())
            if not match:
                continue
            image = cv2.imdecode(
                np.frombuffer(archive.read(entry), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"Cannot decode {entry}")
            scale = min(1.0, 512 / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(
                    image,
                    (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            face = primary_face(reference_detector, image)
            if face is None:
                raise RuntimeError(f"Reference detector missed {entry}")
            landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
            examples.append({
                "file": entry,
                "label": MAPPING[match.group(2)],
                "original": image,
                "glasses": add_glasses(image, landmarks, len(examples)),
            })
    return examples


def evaluate(
    examples: list[dict[str, object]],
    image_key: str,
    threshold: float,
    jpeg_quality: int,
) -> dict[str, object]:
    server.FACE_DETECTION_SCORE_THRESHOLD = threshold
    recognizer = server.EmotionRecognizer()
    rows = []
    for example in examples:
        image = jpeg_roundtrip(example[image_key], jpeg_quality)
        started_at = perf_counter()
        result = recognizer.analyze(image)
        latency_ms = (perf_counter() - started_at) * 1000
        face = result["faces"][0] if result["faces"] else None
        rows.append({
            "file": example["file"],
            "label": example["label"],
            "predicted": face["emotion"]["id"] if face else None,
            "emotion_score": face["emotion"]["score"] if face else None,
            "face_score": face["faceConfidence"] if face else None,
            "latency_ms": latency_ms,
        })
    latencies = np.asarray([row["latency_ms"] for row in rows])
    detected = [row for row in rows if row["predicted"] is not None]
    return {
        "image": image_key,
        "face_threshold": threshold,
        "jpeg_quality": jpeg_quality,
        "examples": len(rows),
        "face_detection_rate": len(detected) / len(rows),
        "accuracy_including_misses": sum(
            row["predicted"] == row["label"] for row in rows
        ) / len(rows),
        "accuracy_when_detected": sum(
            row["predicted"] == row["label"] for row in detected
        ) / max(1, len(detected)),
        "display_rate_at_40_percent": sum(
            row["emotion_score"] is not None and row["emotion_score"] >= 0.4
            for row in rows
        ) / len(rows),
        "mean_face_score": float(np.mean([
            row["face_score"] for row in detected
        ])) if detected else None,
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "records": rows,
    }


def save_sample_grid(examples: list[dict[str, object]], destination: Path) -> None:
    pairs = []
    for example in examples[::20][:7]:
        original = cv2.resize(example["original"], (256, 256))
        glasses = cv2.resize(example["glasses"], (256, 256))
        pairs.append(np.hstack((original, glasses)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", np.vstack(pairs))
    if not success:
        raise RuntimeError("Cannot encode sample grid")
    encoded.tofile(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(r"D:\KDEF\KDEF_and_AKDEF.zip"))
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "glasses-product-benchmark.json")
    parser.add_argument("--model", type=Path, default=server.TRANSFORMER_MODEL_PATH)
    args = parser.parse_args()
    server.CLASSIFIER_SPECS = (
        server.ClassifierSpec("transformer", "ViT-S/16 seven", "Pure ViT", args.model, 7),
    )
    examples = load_examples(args.archive)
    save_sample_grid(examples, ROOT / "output" / "glasses-synthetic-samples.jpg")
    scenarios = {
        "original_previous": ("original", 0.65, 76),
        "original_updated": ("original", 0.50, 85),
        "glasses_previous": ("glasses", 0.65, 76),
        "glasses_threshold_only": ("glasses", 0.50, 76),
        "glasses_updated": ("glasses", 0.50, 85),
    }
    reports = {
        name: evaluate(examples, *configuration)
        for name, configuration in scenarios.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": str(args.model),
        "scope": "140 identity-disjoint KDEF frontal photographs with landmark-placed synthetic glasses and glare; not measured on real eyewear or a live laboratory camera",
        "scenarios": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {key: value for key, value in report.items() if key != "records"}
        for name, report in reports.items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
