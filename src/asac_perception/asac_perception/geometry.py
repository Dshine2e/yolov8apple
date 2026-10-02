"""Optical-axis depth geometry and approximate sphere measurements; no ROS or truth."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class GeometryConfig:
    min_mask_pixels: int = 100
    hole_max_pixels: int = 8
    min_solidity: float = 0.88
    erosion_pixels: int = 2
    min_depth_m: float = 0.15
    max_depth_m: float = 3.0
    depth_mad_factor: float = 4.5
    depth_mad_floor_m: float = 0.008
    min_depth_points: int = 80
    min_depth_ratio: float = 0.3
    max_fit_points: int = 2000
    sphere_trials: int = 80
    sphere_inlier_threshold_m: float = 0.003
    sphere_min_inlier_ratio: float = 0.7
    sphere_max_rmse_m: float = 0.003
    sphere_min_radius_m: float = 0.015
    sphere_max_radius_m: float = 0.09
    sphere_max_condition: float = 100.0
    sphere_min_cap_fraction: float = 0.12


def silhouette(mask, config):
    binary = np.asarray(mask, np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary.astype(bool), {"mask_valid": False, "reason": "empty_mask"}
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = (labels == largest).astype(np.uint8)
    discarded = int(binary.sum() - clean.sum())
    # Fill only bounded small holes; no smoothing, dilation, or circle replacement.
    n, holes, hstats, _ = cv2.connectedComponentsWithStats(1 - clean, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = hstats[i]
        if (
            0 < x
            and 0 < y
            and x + w < clean.shape[1]
            and y + h < clean.shape[0]
            and area <= config.hole_max_pixels
        ):
            clean[holes == i] = 1
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    # Include inner hole boundaries in P when holes survive cleanup.
    all_contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in all_contours)
    area = int(clean.sum())
    border = bool(clean[0].any() or clean[-1].any() or clean[:, 0].any() or clean[:, -1].any())
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / hull_area if hull_area else 0.0
    reason = "image_boundary" if border else ("small_mask" if area < config.min_mask_pixels else "")
    return clean.astype(bool), {
        "mask_valid": not bool(reason),
        "reason": reason,
        "area_px": area,
        "perimeter_px": perimeter,
        "radius_px": math.sqrt(area / math.pi),
        "circularity_top": 4 * math.pi * area / perimeter**2 if perimeter > 0 else None,
        "border_clipped": border,
        "discarded_component_pixels": discarded,
        "solidity": solidity,
    }


def depth_meters(image, encoding, scale_16=0.001):
    if image.ndim != 2:
        raise ValueError("depth_not_2d")
    if encoding == "16UC1" and image.dtype == np.uint16 and 0 < scale_16 <= 1:
        return image.astype(np.float64) * scale_16
    if encoding == "32FC1" and image.dtype == np.float32:
        return image.astype(np.float64)
    raise ValueError("unsupported_depth_encoding_or_dtype")


def project_pixels(u, v, z, k, d=(), model="plumb_bob"):
    k = np.asarray(k, float).reshape(3, 3)
    d = np.asarray(d, float)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("uncalibrated_intrinsics")
    if not np.isfinite(d).all():
        raise ValueError("nonfinite_distortion")
    if model not in ("", "plumb_bob", "rational_polynomial", "equidistant"):
        raise ValueError("unsupported_distortion_model")
    pixels = np.column_stack((u, v)).astype(float).reshape(-1, 1, 2)
    if d.size and np.any(d):
        if model == "equidistant" and d.size == 4:
            rays = cv2.fisheye.undistortPoints(pixels, k, d).reshape(-1, 2)
        elif model in ("plumb_bob", "rational_polynomial") and d.size in (4, 5, 8, 12, 14):
            rays = cv2.undistortPoints(pixels, k, d).reshape(-1, 2)
        else:
            raise ValueError("invalid_distortion_coefficients")
    else:
        rays = (pixels.reshape(-1, 2) - k[:2, 2]) / [k[0, 0], k[1, 1]]
    points = np.column_stack((rays * np.asarray(z)[:, None], z))
    if not np.isfinite(points).all():
        raise ValueError("nonfinite_projection")
    return points


def mask_points(mask, depth, k, d, model, c):
    if depth.shape != mask.shape:
        raise ValueError("unaligned_depth_resolution")
    eroded = mask.astype(np.uint8)
    if c.erosion_pixels:
        size = 2 * c.erosion_pixels + 1
        eroded = cv2.erode(eroded, np.ones((size, size), np.uint8))
    v, u = np.nonzero(eroded)
    z = depth[v, u]
    valid = np.isfinite(z) & (z >= c.min_depth_m) & (z <= c.max_depth_m)
    if valid.any():
        median = np.median(z[valid])
        mad = 1.4826 * np.median(np.abs(z[valid] - median))
        valid &= np.abs(z - median) <= max(c.depth_mad_floor_m, c.depth_mad_factor * mad)
    count = int(valid.sum())
    ratio = count / max(1, len(z))  # denominator is eroded silhouette pixels
    points = project_pixels(u[valid], v[valid], z[valid], k, d, model)
    return points, {
        "valid_depth_points": count,
        "valid_depth_ratio": ratio,
        "depth_candidate_pixels": len(z),
    }


def _algebraic_sphere(points):
    origin = points.mean(axis=0)
    p = points - origin
    a = np.column_stack((2 * p, np.ones(len(p))))
    x, _, rank, _ = np.linalg.lstsq(a, np.sum(p * p, axis=1), rcond=None)
    if rank != 4:
        return None
    radius2 = x[3] + np.dot(x[:3], x[:3])
    if radius2 <= 0:
        return None
    return np.r_[x[:3] + origin, math.sqrt(radius2)]


def fit_sphere(points, c):
    out = {
        "center_camera": None,
        "radius_m": None,
        "diameter_m": None,
        "fit_valid": False,
        "method": "robust_sphere_visible_surface",
        "fit_rmse_m": None,
        "fit_condition": None,
        "inlier_ratio": None,
        "cap_fraction": None,
        "reason": "insufficient_depth",
    }
    if len(points) < c.min_depth_points:
        return out
    rng = np.random.default_rng(42)
    p = points[rng.choice(len(points), min(len(points), c.max_fit_points), replace=False)]
    best, best_count = None, 0
    for i in range(c.sphere_trials + 1):
        estimate = _algebraic_sphere(p if i == 0 else p[rng.choice(len(p), 4, replace=False)])
        if estimate is None or not c.sphere_min_radius_m <= estimate[3] <= c.sphere_max_radius_m:
            continue
        residual = np.abs(np.linalg.norm(p - estimate[:3], axis=1) - estimate[3])
        inliers = residual <= c.sphere_inlier_threshold_m
        if int(inliers.sum()) > best_count:
            best, best_count = (estimate, inliers), int(inliers.sum())
    if best is None or best_count < c.min_depth_points:
        out["reason"] = "sphere_no_consensus"
        return out
    estimate, inliers = best

    def f(x):
        return np.linalg.norm(p[inliers] - x[:3], axis=1) - x[3]

    result = least_squares(
        f, estimate, loss="soft_l1", f_scale=c.sphere_inlier_threshold_m, max_nfev=100
    )
    center, radius = result.x[:3], float(result.x[3])
    residual = np.abs(np.linalg.norm(p - center, axis=1) - radius)
    inliers = residual <= c.sphere_inlier_threshold_m
    q = p[inliers] - center
    if len(q) < c.min_depth_points or radius <= 0:
        out["reason"] = "sphere_no_consensus"
        return out
    normals = q / np.linalg.norm(q, axis=1)[:, None]
    singular = np.linalg.svd(np.column_stack((normals, np.ones(len(q)))), compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    # Cap height / diameter along camera view. Shallow patches are ill-constrained.
    axis = center / np.linalg.norm(center)
    cap = float(np.ptp(q @ axis) / (2 * radius))
    ratio = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    out.update(fit_rmse_m=rmse, fit_condition=condition, inlier_ratio=ratio, cap_fraction=cap)
    reason = ""
    if not result.success or not np.isfinite(result.x).all():
        reason = "sphere_optimizer_failed"
    elif not c.sphere_min_radius_m <= radius <= c.sphere_max_radius_m:
        reason = "sphere_radius_out_of_range"
    elif ratio < c.sphere_min_inlier_ratio or rmse > c.sphere_max_rmse_m:
        reason = "sphere_residual_or_inliers"
    elif condition > c.sphere_max_condition or cap < c.sphere_min_cap_fraction:
        reason = "sphere_ill_conditioned_partial_surface"
    elif center[2] <= np.median(points[:, 2]):
        reason = "sphere_center_on_wrong_side"
    out["reason"] = reason
    if not reason:
        out.update(
            fit_valid=True, center_camera=center.tolist(), radius_m=radius, diameter_m=2 * radius
        )
    return out


def transform_point(point, translation, quaternion_xyzw):
    t, q = np.asarray(translation, float), np.asarray(quaternion_xyzw, float)
    if not np.isfinite(t).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("invalid_tf")
    return (Rotation.from_quat(q).apply(point) + t).tolist()


def observation_direction(center, camera_position):
    vector = np.asarray(camera_position) - np.asarray(center)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("undefined_observation_direction")
    return (vector / norm).tolist()
