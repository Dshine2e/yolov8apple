"""Sensor renderer and URDF tests, no claim of learned recognition."""

from pathlib import Path

import numpy as np
import yaml

from asac_sim.scene import Scene, robot_description


def test_official_urdf_root_and_resource_uris():
    assets = Path(__file__).parents[1] / "assets/piper"
    text, root = robot_description(assets, for_rviz=True)
    assert root == "base_link"  # asserts verified upstream contract, not perception default
    assert "file://" in text and "package://piper_description" not in text


def test_actual_depth_rendering_registration_and_optical_axis():
    package = Path(__file__).parents[1]
    config = yaml.safe_load((package / "config/scene.yaml").read_text())
    config["apples"] = []
    config["occluders"] = []
    scene = Scene(config, package / "assets/piper")
    try:
        rgb, depth = scene.render()
        assert rgb.shape == (*depth.shape, 3) == (480, 640, 3)
        assert depth.dtype == np.float32 and rgb.dtype == np.uint8
        # Center pixel sees tray top z=.021 from camera z=.85.
        assert abs(float(depth[240, 320]) - (0.85 - 0.021)) < 0.002
        np.testing.assert_allclose(scene.rotation.apply([0, 0, 1]), [0, 0, -1], atol=1e-12)
    finally:
        scene.close()
