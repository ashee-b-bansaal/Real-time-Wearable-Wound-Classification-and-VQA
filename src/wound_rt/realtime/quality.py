from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_laplacian_var: float = 80.0
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    min_contrast_std: float = 20.0
    max_motion_mad: float = 16.0


@dataclass
class QualityStats:
    laplacian_var: float
    brightness: float
    contrast_std: float
    motion_mad: float
    pass_quality: bool


def compute_quality(frame_bgr: np.ndarray, prev_gray: np.ndarray | None, thr: QualityThresholds) -> QualityStats:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast_std = float(gray.std())
    if prev_gray is None:
        motion_mad = 0.0
    else:
        motion_mad = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))

    pass_quality = (
        lap_var >= thr.min_laplacian_var
        and thr.min_brightness <= brightness <= thr.max_brightness
        and contrast_std >= thr.min_contrast_std
        and motion_mad <= thr.max_motion_mad
    )
    return QualityStats(
        laplacian_var=lap_var,
        brightness=brightness,
        contrast_std=contrast_std,
        motion_mad=motion_mad,
        pass_quality=pass_quality,
    )
