from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import onnxruntime as ort

from training.emotion_preprocessing import emotion_tensor_bgr


ROOT = Path(__file__).resolve().parent
APPLICATION_ID = "seven-class-facial-expression-system"
INSTANCE_ID = hashlib.sha256(str(ROOT).casefold().encode("utf-8")).hexdigest()[:16]
WEB_ROOT = ROOT / "web"
TRANSFORMER_MODEL_PATH = (
    ROOT / "models" / "vit_emotion_seven_fp32.onnx"
)
FACE_DETECTOR_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
FEEDBACK_ROOT = ROOT / "data" / "custom" / "feedback"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_MODEL_ID = "transformer"
FACE_DETECTION_SCORE_THRESHOLD = 0.50

EMOTIONS = (
    ("anger", "愤怒"),
    ("disgust", "厌恶"),
    ("fear", "恐惧"),
    ("happiness", "开心"),
    ("neutral", "中性"),
    ("sadness", "悲伤"),
    ("surprise", "惊讶"),
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

FaceCandidate = tuple[int, int, int, int, float, np.ndarray]


@dataclass(frozen=True)
class ClassifierSpec:
    id: str
    name: str
    architecture: str
    path: Path
    emotion_count: int


CLASSIFIER_SPECS = (
    ClassifierSpec(
        id="transformer",
        name="Custom ViT-S/16 Seven-Class FP32 (20260928)",
        architecture="Pure Vision Transformer (ViT-S/16)",
        path=TRANSFORMER_MODEL_PATH,
        emotion_count=7,
    ),
)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


class EmotionRecognizer:
    def __init__(self, face_detector_path: Path = FACE_DETECTOR_PATH) -> None:
        if not face_detector_path.is_file():
            raise FileNotFoundError(
                f"Missing face detector: {face_detector_path}. Run scripts/download-model.ps1 first."
            )
        opencv_face_detector_path = self._opencv_safe_path(face_detector_path)
        self.face_detector = cv2.FaceDetectorYN.create(
            str(opencv_face_detector_path),
            "",
            (320, 320),
            FACE_DETECTION_SCORE_THRESHOLD,
            0.3,
            5000,
        )
        self.inference_lock = Lock()
        self.classifiers: dict[str, dict[str, Any]] = {}
        for spec in CLASSIFIER_SPECS:
            if not spec.path.is_file():
                raise FileNotFoundError(
                    f"Missing model: {spec.path}. Restore it from the deployment package; "
                    "the custom ViT is not available from the public model downloader."
                )
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                str(spec.path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self.classifiers[spec.id] = {
                "spec": spec,
                "session": session,
                "input_name": session.get_inputs()[0].name,
                "output_name": session.get_outputs()[0].name,
                "input_shape": list(session.get_inputs()[0].shape),
                "output_shape": list(session.get_outputs()[0].shape),
            }

    def model_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "name": runtime["spec"].name,
                "architecture": runtime["spec"].architecture,
                "inputShape": runtime["input_shape"],
                "outputShape": runtime["output_shape"],
                "displayClasses": [label for label, _ in EMOTIONS],
                "displayOutputShape": ["batch_size", len(EMOTIONS)],
                "preprocessing": "grayscale-square-face-crop",
                "provider": runtime["session"].get_providers()[0],
            }
            for model_id, runtime in self.classifiers.items()
        ]

    @staticmethod
    def _opencv_safe_path(model_path: Path) -> Path:
        try:
            str(model_path).encode("ascii")
            return model_path
        except UnicodeEncodeError:
            cache_directory = Path(tempfile.gettempdir()) / "emotion-recognition-demo"
            cache_directory.mkdir(parents=True, exist_ok=True)
            cached_model = cache_directory / model_path.name
            if (
                not cached_model.is_file()
                or cached_model.stat().st_size != model_path.stat().st_size
                or cached_model.stat().st_mtime_ns < model_path.stat().st_mtime_ns
            ):
                shutil.copy2(model_path, cached_model)
            return cached_model

    @staticmethod
    def _square_face_box(
        x: int,
        y: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        side = int(round(max(width, height) * 1.45))
        center_x = x + width / 2
        center_y = y + height / 2
        left = max(0, int(round(center_x - side / 2)))
        top = max(0, int(round(center_y - side / 2)))
        right = min(image_width, left + side)
        bottom = min(image_height, top + side)
        left = max(0, right - side)
        top = max(0, bottom - side)
        return left, top, right - left, bottom - top

    @staticmethod
    def _intersection_over_union(
        first: FaceCandidate,
        second: FaceCandidate,
    ) -> float:
        first_x, first_y, first_width, first_height = first[:4]
        second_x, second_y, second_width, second_height = second[:4]
        intersection_left = max(first_x, second_x)
        intersection_top = max(first_y, second_y)
        intersection_right = min(first_x + first_width, second_x + second_width)
        intersection_bottom = min(first_y + first_height, second_y + second_height)
        intersection_width = max(0, intersection_right - intersection_left)
        intersection_height = max(0, intersection_bottom - intersection_top)
        intersection_area = intersection_width * intersection_height
        union_area = (
            first_width * first_height
            + second_width * second_height
            - intersection_area
        )
        return intersection_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _intersection_over_smaller(
        first: FaceCandidate,
        second: FaceCandidate,
    ) -> float:
        first_x, first_y, first_width, first_height = first[:4]
        second_x, second_y, second_width, second_height = second[:4]
        intersection_width = max(
            0,
            min(first_x + first_width, second_x + second_width)
            - max(first_x, second_x),
        )
        intersection_height = max(
            0,
            min(first_y + first_height, second_y + second_height)
            - max(first_y, second_y),
        )
        smaller_area = min(first_width * first_height, second_width * second_height)
        return (
            intersection_width * intersection_height / smaller_area
            if smaller_area > 0
            else 0.0
        )

    @staticmethod
    def _normalized_center_distance(
        first: FaceCandidate,
        second: FaceCandidate,
    ) -> float:
        first_x, first_y, first_width, first_height = first[:4]
        second_x, second_y, second_width, second_height = second[:4]
        first_center = np.asarray(
            (first_x + first_width / 2, first_y + first_height / 2),
            dtype=np.float32,
        )
        second_center = np.asarray(
            (second_x + second_width / 2, second_y + second_height / 2),
            dtype=np.float32,
        )
        reference_side = max(
            1.0,
            min(max(first_width, first_height), max(second_width, second_height)),
        )
        return float(np.linalg.norm(first_center - second_center) / reference_side)

    @classmethod
    def _same_face_candidate(
        cls,
        first: FaceCandidate,
        second: FaceCandidate,
    ) -> bool:
        if cls._intersection_over_union(first, second) >= 0.32:
            return True
        if cls._intersection_over_smaller(first, second) >= 0.62:
            return True
        first_area = first[2] * first[3]
        second_area = second[2] * second[3]
        area_ratio = max(first_area, second_area) / max(1, min(first_area, second_area))
        return area_ratio <= 2.25 and cls._normalized_center_distance(first, second) <= 0.28

    @classmethod
    def _deduplicate_faces(
        cls,
        candidates: list[FaceCandidate],
    ) -> list[FaceCandidate]:
        selected: list[FaceCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
            if all(not cls._same_face_candidate(candidate, kept) for kept in selected):
                selected.append(candidate)
        return selected

    @staticmethod
    def _emotion_tensor(
        face_bgr: np.ndarray,
    ) -> np.ndarray:
        return emotion_tensor_bgr(
            face_bgr,
            grayscale=True,
            evaluation_zoom=1.0,
            add_batch_dimension=True,
        )

    def _detect_faces_unlocked(self, image: np.ndarray) -> list[FaceCandidate]:
        image_height, image_width = image.shape[:2]
        self.face_detector.setInputSize((image_width, image_height))
        _, raw_faces = self.face_detector.detect(image)
        candidates = [] if raw_faces is None else [
            (
                int(round(face[0])), int(round(face[1])),
                int(round(face[2])), int(round(face[3])), float(face[14]),
                np.asarray(face[4:14], dtype=np.float32).reshape(5, 2),
            )
            for face in raw_faces
            if face[2] >= 24 and face[3] >= 24
        ]
        return self._deduplicate_faces(candidates)

    def _detect_faces(self, image: np.ndarray) -> list[FaceCandidate]:
        with self.inference_lock:
            candidates = self._detect_faces_unlocked(image)
            largest_confident_area = max(
                (face[2] * face[3] for face in candidates if face[4] >= 0.9),
                default=0,
            )
            if len(candidates) < 2 or not largest_confident_area:
                return candidates

            image_height, image_width = image.shape[:2]
            accepted = []
            for face in candidates:
                x, y, width, height, score, _ = face
                # Recheck large weak boxes that can displace a convincing face.
                # Ordinary single faces and smaller distant faces keep the
                # original low detection threshold, including eyewear cases.
                if score >= 0.9 or width * height < 1.5 * largest_confident_area:
                    accepted.append(face)
                    continue
                left, top, crop_width, crop_height = self._square_face_box(
                    x, y, width, height, image_width, image_height,
                )
                crop = image[top : top + crop_height, left : left + crop_width]
                if not crop.size:
                    continue
                interpolation = cv2.INTER_AREA if min(crop.shape[:2]) >= 224 else cv2.INTER_LINEAR
                crop = cv2.resize(crop, (224, 224), interpolation=interpolation)
                secondary = self._detect_faces_unlocked(crop)
                if any(
                    confidence >= 0.85
                    and abs((sx + sw / 2) / 224 - 0.5) < 0.3
                    and abs((sy + sh / 2) / 224 - 0.5) < 0.3
                    and sw * sh / 224**2 >= 0.08
                    for sx, sy, sw, sh, confidence, _ in secondary
                ):
                    accepted.append(face)
            return accepted

    def analyze(self, image: np.ndarray, model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
        if model_id not in self.classifiers:
            raise ValueError(f"Unknown emotion model: {model_id}")
        classifier = self.classifiers[model_id]
        spec: ClassifierSpec = classifier["spec"]
        started_at = perf_counter()
        classifier_seconds = 0.0
        image_height, image_width = image.shape[:2]
        detection_started_at = perf_counter()
        faces = self._detect_faces(image)
        detection_seconds = perf_counter() - detection_started_at
        faces.sort(key=lambda item: item[2] * item[3], reverse=True)
        predictions: list[dict[str, Any]] = []

        for raw_box in faces[:4]:
            x, y, width, height, detection_score, landmarks = raw_box
            left, top, box_width, box_height = self._square_face_box(
                x,
                y,
                width,
                height,
                image_width,
                image_height,
            )
            classification_face = image[
                top : top + box_height,
                left : left + box_width,
            ]
            if classification_face.size == 0:
                continue

            tensor = self._emotion_tensor(classification_face)
            classifier_started_at = perf_counter()
            with self.inference_lock:
                raw_scores = classifier["session"].run(
                    [classifier["output_name"]],
                    {classifier["input_name"]: tensor},
                )[0]
            classifier_seconds += perf_counter() - classifier_started_at
            raw_logits = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
            emotion_logits = raw_logits[: spec.emotion_count]
            probabilities = softmax(emotion_logits)
            ranked_indices = np.argsort(probabilities)[::-1]
            scores = [
                {
                    "id": EMOTIONS[index][0],
                    "label": EMOTIONS[index][1],
                    "score": round(float(probabilities[index]), 6),
                }
                for index in ranked_indices
            ]
            predictions.append(
                {
                    "bbox": {
                        # Draw the detector's tight face rectangle in the UI.
                        # The padded square above remains the classifier input so
                        # the visual change does not alter model predictions.
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    },
                    "cropBbox": {
                        "x": left,
                        "y": top,
                        "width": box_width,
                        "height": box_height,
                    },
                    "emotion": scores[0],
                    "scores": scores,
                    "faceConfidence": round(detection_score, 6),
                    "aligned": False,
                }
            )

        return {
            "model": {"id": spec.id, "name": spec.name},
            "width": image_width,
            "height": image_height,
            "faces": predictions,
            "latencyMs": round((perf_counter() - started_at) * 1000, 1),
            "timingMs": {
                "detection": round(detection_seconds * 1000, 1),
                "classification": round(classifier_seconds * 1000, 1),
            },
        }

    def feedback_face(self, image: np.ndarray) -> dict[str, Any] | None:
        image_height, image_width = image.shape[:2]
        faces = self._detect_faces(image)
        if not faces:
            return None
        x, y, width, height, confidence, _ = max(
            faces,
            key=lambda item: item[2] * item[3],
        )
        left, top, crop_width, crop_height = self._square_face_box(
            x,
            y,
            width,
            height,
            image_width,
            image_height,
        )
        crop = image[top : top + crop_height, left : left + crop_width]
        if crop.size == 0:
            return None
        interpolation = (
            cv2.INTER_AREA
            if crop_width >= 224 and crop_height >= 224
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(crop, (224, 224), interpolation=interpolation)
        return {
            "image": resized,
            "bbox": {
                "x": left,
                "y": top,
                "width": crop_width,
                "height": crop_height,
            },
            "faceConfidence": confidence,
            "sourceWidth": image_width,
            "sourceHeight": image_height,
        }


class ApplicationHandler(SimpleHTTPRequestHandler):
    recognizer: EmotionRecognizer

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            models = self.recognizer.model_metadata()
            default_model = next(
                model for model in models if model["id"] == DEFAULT_MODEL_ID
            )
            self._send_json(
                200,
                {
                    "ready": True,
                    "application": APPLICATION_ID,
                    "instance": INSTANCE_ID,
                    "model": default_model["name"],
                    "detector": "YuNet 2023",
                    "defaultModel": DEFAULT_MODEL_ID,
                    "models": models,
                    "provider": default_model["provider"],
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path not in ("/api/analyze", "/api/feedback"):
            self._send_json(404, {"error": "Not found"})
            return
        query = parse_qs(parsed_url.query)
        if parsed_url.path == "/api/analyze":
            model_id = query.get("model", [DEFAULT_MODEL_ID])[0]
            if model_id not in self.recognizer.classifiers:
                self._send_json(400, {"error": f"Unknown emotion model: {model_id}"})
                return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(400, {"error": "Image payload is empty or too large"})
            return

        try:
            encoded = np.frombuffer(self.rfile.read(content_length), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unable to decode image")
            if parsed_url.path == "/api/analyze":
                self._send_json(200, self.recognizer.analyze(image, model_id))
                return

            valid_labels = {emotion_id for emotion_id, _ in EMOTIONS}
            label = query.get("label", [""])[0]
            if label not in valid_labels:
                self._send_json(400, {"error": f"Unknown feedback label: {label}"})
                return
            session = re.sub(
                r"[^A-Za-z0-9_-]",
                "_",
                query.get("session", ["natural"])[0],
            )[:64] or "natural"
            predicted = query.get("predicted", [""])[0]
            predicted_score = query.get("score", [""])[0]
            feedback_model = query.get("model", [DEFAULT_MODEL_ID])[0]
            feedback = self.recognizer.feedback_face(image)
            if feedback is None:
                self._send_json(400, {"error": "No face found in feedback frame"})
                return

            captured_at = datetime.now().astimezone()
            label_root = FEEDBACK_ROOT / session / label
            label_root.mkdir(parents=True, exist_ok=True)
            filename = captured_at.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            output_path = label_root / filename
            success, jpeg = cv2.imencode(
                ".jpg",
                feedback["image"],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if not success:
                raise RuntimeError("Unable to encode feedback face")
            jpeg.tofile(str(output_path))
            manifest_path = FEEDBACK_ROOT / "manifest.jsonl"
            record = {
                "file": str(output_path.relative_to(ROOT)),
                "session": session,
                "label": label,
                "predicted": predicted,
                "predictedScore": predicted_score,
                "model": feedback_model,
                "capturedAt": captured_at.isoformat(),
                "bbox": feedback["bbox"],
                "faceConfidence": round(float(feedback["faceConfidence"]), 6),
                "sourceWidth": feedback["sourceWidth"],
                "sourceHeight": feedback["sourceHeight"],
                "cropMethod": "yunet-square-1.45",
            }
            with manifest_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._send_json(
                200,
                {
                    "saved": True,
                    "label": label,
                    "session": session,
                    "count": len(list(label_root.glob("*.jpg"))),
                },
            )
        except Exception as error:
            self._send_json(500, {"error": str(error)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local seven-class facial-expression recognition system")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimetypes.add_type("text/javascript", ".js")
    ApplicationHandler.recognizer = EmotionRecognizer()
    server = HTTPServer((args.host, args.port), ApplicationHandler)
    print(f"Seven-class facial-expression recognition system: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping service...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
