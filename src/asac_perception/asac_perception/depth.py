"""ROS 의존성이 없는 깊이 추정 및 optical 좌표 역투영."""

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthEstimate:
    valid: bool = False
    samples: int = 0
    x: float = math.nan
    y: float = math.nan
    z: float = math.nan

    @property
    def range_m(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


def depth_in_meters(image: np.ndarray, encoding: str) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError('Depth image must be a two-dimensional array')
    if encoding == '16UC1':
        return image.astype(np.float32) * 0.001  # ROS RealSense: mm -> m
    if encoding == '32FC1':
        return image.astype(np.float32, copy=False)  # ROS: m
    raise ValueError(f'Unsupported depth encoding: {encoding}')


def normalized_ray(
    u: float, v: float, k: Sequence[float], d: Sequence[float], model: str,
) -> tuple[float, float]:
    matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(d, dtype=np.float64)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError('CameraInfo must contain finite, positive focal lengths')
    if not np.all(np.isfinite(distortion)):
        raise ValueError('CameraInfo distortion must be finite')
    if model not in ('', 'plumb_bob', 'rational_polynomial', 'equidistant'):
        raise ValueError(f'Unsupported distortion model: {model}')
    if distortion.size == 0 or not np.any(distortion):
        return ((u - matrix[0, 2]) / matrix[0, 0],
                (v - matrix[1, 2]) / matrix[1, 1])
    pixel = np.array([[[u, v]]], dtype=np.float64)
    if model == 'equidistant':
        if distortion.size != 4:
            raise ValueError('Equidistant distortion requires four coefficients')
        ray = cv2.fisheye.undistortPoints(pixel, matrix, distortion)
    else:
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError('Unsupported Brown-Conrady coefficient count')
        ray = cv2.undistortPoints(pixel, matrix, distortion)
    if not np.all(np.isfinite(ray)):
        raise ValueError('Undistortion produced a nonfinite ray')
    return float(ray[0, 0, 0]), float(ray[0, 0, 1])


def estimate_position(
    depth_m: np.ndarray,
    bbox: Sequence[float],
    k: Sequence[float],
    d: Sequence[float] = (),
    distortion_model: str = 'plumb_bob',
    roi_fraction: float = 0.3,
    min_depth_m: float = 0.2,
    max_depth_m: float = 6.0,
    min_samples: int = 5,
) -> DepthEstimate:
    """박스 중앙 ROI의 유효 깊이 중앙값을 박스 중심 광선에 역투영한다."""
    if depth_m.ndim != 2:
        raise ValueError('Depth must be two-dimensional')
    if not 0 < roi_fraction <= 1:
        raise ValueError('roi_fraction must be in (0, 1]')
    if not 0 < min_depth_m < max_depth_m or min_samples < 1:
        raise ValueError('Invalid depth range or min_samples')
    if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
        raise ValueError('bbox must have four finite coordinates')
    height, width = depth_m.shape
    x1, y1, x2, y2 = map(float, bbox)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return DepthEstimate()
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * roi_fraction / 2, (y2 - y1) * roi_fraction / 2
    left, right = max(0, math.floor(u - half_w)), min(width, math.ceil(u + half_w))
    top, bottom = max(0, math.floor(v - half_h)), min(height, math.ceil(v + half_h))
    roi = depth_m[top:bottom, left:right]
    valid = roi[np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)]
    count = int(valid.size)
    if count < min_samples:
        return DepthEstimate(samples=count)
    z = float(np.median(valid))
    ray_x, ray_y = normalized_ray(u, v, k, d, distortion_model)
    return DepthEstimate(True, count, ray_x * z, ray_y * z, z)
