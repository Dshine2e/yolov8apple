import math

import cv2
import numpy as np
import pytest

from asac_perception.geometry import (
    GeometryConfig,
    depth_meters,
    fit_sphere,
    mask_points,
    observation_direction,
    project_pixels,
    silhouette,
    transform_point,
)
from asac_perception.observations import measure_instances
from asac_perception.segmentation import Instance
from asac_perception.tracking import Tracker


def test_projection_and_tf_direction():
    k = [200, 0, 100, 0, 200, 100, 0, 0, 1]
    points = project_pixels([120, 80], [80, 120], [2, 2], k)
    np.testing.assert_allclose(points, [[0.2, -0.2, 2], [-0.2, 0.2, 2]])
    # Camera at z=1 looking down: optical z=.8 maps to base z=.2, y flips.
    transformed = transform_point([0.1, 0.2, 0.8], [0.4, 0, 1], [1, 0, 0, 0])
    np.testing.assert_allclose(transformed, [0.5, -0.2, 0.2])
    d = observation_direction(transformed, [0.4, 0, 1])
    assert np.linalg.norm(d) == pytest.approx(1)
    assert d[2] > 0
    with pytest.raises(ValueError):
        transform_point([1, 2, 3], [0, 0, 0], [0, 0, 0, 0])


def test_raw_distortion_projection():
    k = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], float)
    d = np.array([0.2, -0.05, 0.001, 0.002, 0])
    pixel, _ = cv2.projectPoints(np.array([[0.2, -0.1, 1.0]]), np.zeros(3), np.zeros(3), k, d)
    points = project_pixels([pixel.ravel()[0]], [pixel.ravel()[1]], [1], k, d)
    np.testing.assert_allclose(points, [[0.2, -0.1, 1]], atol=1e-6)
    with pytest.raises(ValueError, match="uncalibrated"):
        project_pixels([0], [0], [1], np.zeros(9))


def test_silhouette_circle_ellipse_and_border():
    c = GeometryConfig()
    circle = np.zeros((256, 256), np.uint8)
    cv2.circle(circle, (128, 128), 60, 1, -1)
    mask, stats = silhouette(circle, c)
    assert stats["mask_valid"]
    assert stats["radius_px"] == pytest.approx(60, abs=0.6)
    assert 0.88 < stats["circularity_top"] < 1.02  # raster perimeter bias is documented
    ellipse = np.zeros_like(circle)
    cv2.ellipse(ellipse, (128, 128), (80, 30), 0, 0, 360, 1, -1)
    _, es = silhouette(ellipse, c)
    assert es["circularity_top"] < stats["circularity_top"] - 0.15
    circle[0:128, 0:128] = 1
    _, stats = silhouette(circle, c)
    assert not stats["mask_valid"] and stats["border_clipped"]
    small = np.zeros((20, 20), np.uint8)
    small[5:8, 5:8] = 1
    assert not silhouette(small, c)[1]["mask_valid"]


def test_holes_limited_cleanup():
    mask = np.zeros((40, 40), np.uint8)
    mask[5:35, 5:35] = 1
    mask[10, 10] = 0
    mask[20:25, 20:25] = 0
    clean, _ = silhouette(mask, GeometryConfig())
    assert clean[10, 10] and not clean[22, 22]


def sphere_cap(seed=7, cap_angle=1.15):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.05, cap_angle, 1500)
    phi = rng.uniform(0, 2 * math.pi, len(theta))
    normals = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta))
    )
    return np.array([0.02, -0.01, 0.8]) + 0.04 * normals, rng


def test_robust_sphere_noisy_visible_cap_outliers():
    points, rng = sphere_cap()
    points += rng.normal(0, 0.0003, points.shape)
    points[:100] += rng.normal(0, 0.02, (100, 3))
    fit = fit_sphere(points, GeometryConfig())
    assert fit["fit_valid"], fit
    np.testing.assert_allclose(fit["center_camera"], [0.02, -0.01, 0.8], atol=0.0015)
    assert fit["radius_m"] == pytest.approx(0.04, abs=0.0015)
    assert fit["diameter_m"] == pytest.approx(0.08, abs=0.003)
    assert fit["inlier_ratio"] > 0.85


def test_planar_and_shallow_surface_are_not_centers():
    points, _ = sphere_cap(cap_angle=0.1)
    fit = fit_sphere(points, GeometryConfig())
    assert not fit["fit_valid"] and fit["center_camera"] is None
    rng = np.random.default_rng(1)
    plane = np.column_stack((rng.uniform(-0.04, 0.04, (1000, 2)), np.full(1000, 0.8)))
    fit = fit_sphere(plane, GeometryConfig())
    assert not fit["fit_valid"] and fit["radius_m"] is None


def test_depth_invalid_units_background_and_roi_alignment():
    c = GeometryConfig()
    bgr = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    mask = np.zeros((60, 80), bool)
    mask[15:40, 20:50] = True
    depth = np.full(mask.shape, 0.7)
    depth[17:20, 22:25] = [0, np.nan, np.inf]
    depth[30:33, 30:33] = 2.0  # tray/background contamination excluded by robust depth gate
    points, stats = mask_points(mask, depth, [100, 0, 40, 0, 100, 30, 0, 0, 1], [], "", c)
    assert stats["valid_depth_ratio"] < 1
    assert np.all(points[:, 2] == 0.7)
    det = Instance(np.array([22, 17, 48, 38]), 0.9, 1, mask)
    observation = measure_instances(bgr, [det], None, ([], [], ""), c)[0]
    x, y, w, h = observation["roi"]
    np.testing.assert_equal(observation["rgb_crop"], bgr[y:y + h, x:x + w])
    np.testing.assert_equal(observation["mask_crop"] > 0, mask[y:y + h, x:x + w])
    assert observation["center_camera"] is None and observation["radius_m"] is None
    assert not observation["direction_valid"]
    assert depth_meters(np.array([[1000]], np.uint16), "16UC1")[0, 0] == 1
    assert depth_meters(np.array([[1.0]], np.float32), "32FC1")[0, 0] == 1
    with pytest.raises(ValueError):
        depth_meters(np.array([[1000]], np.uint16), "32FC1")
    with pytest.raises(ValueError, match="unaligned"):
        mask_points(mask, np.ones((30, 40)), [100, 0, 40, 0, 100, 30, 0, 0, 1], [], "", c)


def test_tracker_permutation_miss_duplicate_add_remove_and_fallback():
    t = Tracker(ttl_sec=2, session="test")

    def o(x, z=True, score=0.9):
        return dict(
            confidence=score,
            bbox=[x, 0, x + 10, 10],
            center_camera=[x / 100, 0, 0.8] if z else None,
        )

    first = t.update([o(0), o(30)], 1, "optical")
    ids = {p["bbox"][0]: p["apple_id"] for p in first}
    second = t.update([o(30), o(0), o(0, score=0.6)], 1.2, "optical")
    assert len(second) == 2
    assert {p["bbox"][0]: p["apple_id"] for p in second} == ids
    t.update([], 1.5, "optical")
    fallback = t.update([o(0, False), o(50)], 2, "optical")
    assert fallback[0]["apple_id"] == ids[0]
    assert fallback[0]["tracking_method"] == "2d_iou_fallback"
    assert fallback[1]["apple_id"] not in ids.values()
    assert t.update([o(30)], 5, "optical")[0]["apple_id"] != ids[30]
    before = t.update([o(30)], 5.1, "optical")[0]["apple_id"]
    assert t.update([o(30)], 5.2, "changed_frame")[0]["apple_id"] != before
    before = t.update([o(30)], 5.3, "changed_frame")[0]["apple_id"]
    assert t.update([o(30)], 1, "changed_frame")[0]["apple_id"] != before
