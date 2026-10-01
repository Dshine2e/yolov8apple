import math

import cv2
import numpy as np
import pytest

from asac_perception.depth import depth_in_meters, estimate_position, normalized_ray


def test_depth_units():
    mm = np.array([[1000, 0, 2500]], dtype=np.uint16)
    np.testing.assert_allclose(depth_in_meters(mm, '16UC1'), [[1.0, 0.0, 2.5]])
    meters = np.array([[1.5, np.nan]], dtype=np.float32)
    np.testing.assert_equal(depth_in_meters(meters, '32FC1'), meters)
    with pytest.raises(ValueError, match='encoding'):
        depth_in_meters(mm, 'mono16')


def test_off_axis_position_and_range():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    k = [100, 0, 50, 0, 100, 50, 0, 0, 1]
    position = estimate_position(depth, [60, 30, 80, 50], k)
    assert position.valid
    assert position.x == pytest.approx(0.4)
    assert position.y == pytest.approx(-0.2)
    assert position.z == pytest.approx(2.0)
    assert position.range_m == pytest.approx(math.sqrt(4.2))


def test_invalid_pixels_and_outlier_do_not_bias_median():
    depth = np.full((10, 10), 1.5, dtype=np.float32)
    depth[3, 3:8] = [0, np.nan, np.inf, 0.1, 10]
    depth[4, 4] = 5.0
    k = [100, 0, 5, 0, 100, 5, 0, 0, 1]
    position = estimate_position(depth, [0, 0, 10, 10], k, roi_fraction=1.0)
    assert position.samples == 95
    assert position.z == pytest.approx(1.5)


def test_insufficient_depth_is_explicitly_invalid():
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[5, 5] = 1.0
    position = estimate_position(depth, [0, 0, 10, 10], np.eye(3).ravel())
    assert not position.valid
    assert position.samples == 1
    assert math.isnan(position.z) and math.isnan(position.range_m)


def test_bbox_clip_and_empty_intersection():
    depth = np.ones((10, 10), dtype=np.float32)
    k = [100, 0, 5, 0, 100, 5, 0, 0, 1]
    assert estimate_position(depth, [-5, -5, 15, 15], k).valid
    assert not estimate_position(depth, [20, 20, 30, 30], k).valid
    with pytest.raises(ValueError, match='finite'):
        estimate_position(depth, [0, 0, np.nan, 5], k)


def test_distortion_is_inverted():
    k = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], dtype=float)
    d = np.array([0.2, -0.05, 0.001, 0.002, 0.0])
    pixel, _ = cv2.projectPoints(
        np.array([[0.2, -0.1, 1.0]]), np.zeros(3), np.zeros(3), k, d,
    )
    ray = normalized_ray(*pixel.ravel(), k.ravel(), d, 'plumb_bob')
    np.testing.assert_allclose(ray, [0.2, -0.1], atol=1e-6)


def test_uncalibrated_camera_is_rejected():
    with pytest.raises(ValueError, match='focal'):
        normalized_ray(0, 0, np.zeros(9), [], 'plumb_bob')
    with pytest.raises(ValueError, match='distortion model'):
        normalized_ray(0, 0, np.eye(3).ravel(), [], 'unknown')
