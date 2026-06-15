from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerConfig:
    marker_width_mm: float = 20.0
    marker_height_mm: float = 20.0
    min_area_px: int = 400


@dataclass
class DimensionResult:
    status: str
    area_mm2: float | None
    major_axis_mm: float | None
    minor_axis_mm: float | None
    px_per_mm: float | None


def detect_reference_marker(frame_bgr: np.ndarray, cfg: MarkerConfig) -> tuple[np.ndarray | None, float | None]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low = np.array([95, 80, 50], dtype=np.uint8)
    high = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < cfg.min_area_px:
        return None, None
    rect = cv2.minAreaRect(best)
    w, h = rect[1]
    if w <= 1 or h <= 1:
        return None, None
    marker_px = (w + h) / 2.0
    marker_mm = (cfg.marker_width_mm + cfg.marker_height_mm) / 2.0
    px_per_mm = marker_px / marker_mm
    box = cv2.boxPoints(rect).astype(np.int32)
    return box, float(px_per_mm)


def rough_wound_mask(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    blur = cv2.GaussianBlur(a, (5, 5), 0)
    _, mask = cv2.threshold(blur, 145, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def estimate_dimensions(frame_bgr: np.ndarray, cfg: MarkerConfig) -> DimensionResult:
    marker_box, px_per_mm = detect_reference_marker(frame_bgr, cfg)
    if px_per_mm is None:
        return DimensionResult(
            status="unavailable_marker_missing",
            area_mm2=None,
            major_axis_mm=None,
            minor_axis_mm=None,
            px_per_mm=None,
        )

    mask = rough_wound_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return DimensionResult(
            status="unavailable_no_wound_contour",
            area_mm2=None,
            major_axis_mm=None,
            minor_axis_mm=None,
            px_per_mm=px_per_mm,
        )

    wound = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(wound)
    if area_px < 50:
        return DimensionResult(
            status="unavailable_small_contour",
            area_mm2=None,
            major_axis_mm=None,
            minor_axis_mm=None,
            px_per_mm=px_per_mm,
        )

    rect = cv2.minAreaRect(wound)
    w_px, h_px = rect[1]
    area_mm2 = area_px / (px_per_mm**2)
    major_axis_mm = max(w_px, h_px) / px_per_mm
    minor_axis_mm = min(w_px, h_px) / px_per_mm

    return DimensionResult(
        status="ok",
        area_mm2=float(area_mm2),
        major_axis_mm=float(major_axis_mm),
        minor_axis_mm=float(minor_axis_mm),
        px_per_mm=float(px_per_mm),
    )
