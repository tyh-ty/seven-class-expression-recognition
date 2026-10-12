from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SIZE = 224
EVALUATION_ZOOM = 1.0
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def emotion_tensor_bgr(
    image_bgr: np.ndarray,
    image_size: int = IMAGE_SIZE,
    grayscale: bool = True,
    evaluation_zoom: float = EVALUATION_ZOOM,
    add_batch_dimension: bool = False,
) -> np.ndarray:
    """Match the exact preprocessing used by the webcam inference path."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected a BGR image with 3 channels, got {image_bgr.shape}")

    source_height, source_width = image_bgr.shape[:2]
    resize_interpolation = (
        cv2.INTER_AREA
        if source_width >= image_size and source_height >= image_size
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        image_bgr,
        (image_size, image_size),
        interpolation=resize_interpolation,
    )
    if evaluation_zoom > 1.0:
        enlarged_size = int(round(image_size * evaluation_zoom))
        enlarged = cv2.resize(
            resized,
            (enlarged_size, enlarged_size),
            interpolation=cv2.INTER_LINEAR,
        )
        offset = (enlarged_size - image_size) // 2
        resized = enlarged[offset : offset + image_size, offset : offset + image_size]

    if grayscale:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = normalized.transpose(2, 0, 1).astype(np.float32)
    if add_batch_dimension:
        tensor = tensor[np.newaxis, ...]
    return tensor


def emotion_tensor_path(
    path: Path,
    image_size: int = IMAGE_SIZE,
    grayscale: bool = True,
    evaluation_zoom: float = EVALUATION_ZOOM,
) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to decode image: {path}")
    return emotion_tensor_bgr(
        image_bgr,
        image_size=image_size,
        grayscale=grayscale,
        evaluation_zoom=evaluation_zoom,
    )


class EmotionEvaluationTransform:
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        grayscale: bool = True,
        evaluation_zoom: float = EVALUATION_ZOOM,
    ) -> None:
        self.image_size = image_size
        self.grayscale = grayscale
        self.evaluation_zoom = evaluation_zoom

    def __call__(self, image: Any) -> Any:
        import torch

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tensor = emotion_tensor_bgr(
            bgr,
            image_size=self.image_size,
            grayscale=self.grayscale,
            evaluation_zoom=self.evaluation_zoom,
        )
        return torch.from_numpy(tensor)
