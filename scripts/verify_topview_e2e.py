#!/usr/bin/env python3
"""Actual-model ROS scenario runner. Use an isolated ROS_DOMAIN_ID, never hardware."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from visualization_msgs.msg import MarkerArray
import yaml

from asac_interfaces.msg import AppleObservationArray
from asac_sim.evaluation import compare, TOLERANCES


class Probe(Node):
    def __init__(self):
        super().__init__("e2e_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.observations, self.truth, self.statuses = [], {}, []
        self.arrays, self.overlays, self.markers, self.errors = [], [], [], []
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.create_subscription(
            String,
            "/asac/observations_json",
            lambda m: self.observations.append(json.loads(m.data)),
            10,
        )
        self.create_subscription(
            String, "/asac/status", lambda m: self.statuses.append(json.loads(m.data)["state"]), 10
        )
        self.create_subscription(String, "/asac_sim/evaluation_truth", self._truth, 50)
        self.create_subscription(AppleObservationArray, "/asac/observations", self._array, 5)
        self.create_subscription(Image, "/asac/overlay", self.overlays.append, 1)
        self.create_subscription(MarkerArray, "/asac/markers", self.markers.append, 1)

    def _truth(self, msg):
        payload = json.loads(msg.data)
        s = payload["stamp"]
        self.truth[(s["sec"], s["nanosec"])] = payload

    def _array(self, msg):
        self.arrays.append(msg)
        for o in msg.apples:
            if (
                o.header != msg.header
                or o.rgb_crop.header != msg.header
                or o.mask_crop.header != msg.header
            ):
                self.errors.append("crop_capture_header_mismatch")
            if (o.rgb_crop.width, o.rgb_crop.height) != (o.roi.width, o.roi.height) or (
                o.mask_crop.width,
                o.mask_crop.height,
            ) != (o.roi.width, o.roi.height):
                self.errors.append("crop_roi_size_mismatch")
            if o.mask_crop.encoding != "mono8":
                self.errors.append("mask_encoding_mismatch")
            if o.center_base_valid and o.center_base.header.frame_id != msg.base_frame:
                self.errors.append("base_frame_mismatch")


def scenarios(base):
    cases = []

    def add(name, changes=None, expected="accuracy"):
        c = copy.deepcopy(base)
        if changes:
            changes(c)
        cases.append((name, c, expected))

    add("baseline")
    add("single_apple", lambda c: c.update(apples=[c["apples"][1]]))
    add("sphere", lambda c: c.update(apples=[dict(c["apples"][1], shape="sphere")]))

    def changed(c):
        c["apples"][1].update(position=[0.47, 0.08, 0.069], radius_m=0.050)
        c["apples"][0]["position"] = [0.31, -0.11, 0.064]

    add("position_size_change", changed)

    def overlap(c):
        c["apples"][0]["position"] = [0.51, 0.07, 0.066]

    add("overlap", overlap, "report")

    def occlude(c):
        c["occluders"] = [{"position": [0.48, 0.075, 0.12], "size": [0.035, 0.10, 0.012]}]

    add("occlusion", occlude, "report")

    def lifecycle(c):
        c["events"] = [
            {"time_sec": 5, "action": "remove", "name": "red_apple"},
            {"time_sec": 7, "action": "add", "apple": dict(c["apples"][0], name="new_apple")},
        ]

    add("add_remove", lifecycle, "report")
    add("missed_sensor_frames", lambda c: c["faults"].update(drop_rgb_every=3))
    add("depth_holes", lambda c: c["faults"].update(depth_hole_fraction=1.0), "invalid_geometry")
    add(
        "uncalibrated_intrinsics",
        lambda c: c["faults"].update(uncalibrated_info=True),
        "invalid_geometry",
    )
    add(
        "depth_unit_error",
        lambda c: c["faults"].update(depth_unit_multiplier=1000.0),
        "invalid_geometry",
    )
    add("missing_tf", lambda c: c["faults"].update(disable_camera_tf=True), "invalid_base")
    add("delayed_input", lambda c: c["faults"].update(stamp_delay_sec=3.0), "stale")
    add("changed_frame", lambda c: c["faults"].update(change_frame_after_sec=5.0), "changed_frame")
    return cases


def summarize(probe, expected):
    evaluations = []
    all_apples = [o for p in probe.observations for o in p["observations"]]
    switches, previous = 0, {}
    for payload in probe.observations:
        s = payload["stamp"]
        truth = probe.truth.get((s["sec"], s["nanosec"]))
        if truth:
            result = compare(payload["observations"], truth["apples"])
            evaluations.append(result)
            for match in result["matches"]:
                name, aid = match["truth_name"], match["apple_id"]
                if name in previous and previous[name] != aid:
                    switches += 1
                previous[name] = aid
    truth_count = sum(e["truth_count"] for e in evaluations)
    matches = [m for e in evaluations for m in e["matches"]]
    match_ratio = len(matches) / truth_count if truth_count else 0
    measured = len(all_apples)
    check = False
    if expected == "accuracy":
        check = (
            len(evaluations) >= 3
            and match_ratio >= TOLERANCES["min_match_ratio"]
            and all(m["within_tolerance"] for m in matches)
            and switches == 0
        )
    elif expected == "invalid_geometry":
        check = measured > 0 and all(
            o["center_camera"] is None
            and o["radius_m"] is None
            and not o["center_base_valid"]
            and not o["direction_valid"]
            and bool(o["reasons"])
            for o in all_apples
        )
    elif expected == "invalid_base":
        check = measured > 0 and all(
            o["center_base"] is None
            and not o["direction_valid"]
            and any("tf_" in r for r in o["reasons"])
            for o in all_apples
        )
    elif expected == "stale":
        check = not probe.observations and "stale_or_future_input" in probe.statuses
    elif expected == "changed_frame":
        changed = [
            o
            for p in probe.observations
            if p["camera_frame"].endswith("_changed")
            for o in p["observations"]
        ]
        check = bool(changed) and all(
            o["center_base"] is None and o["radius_m"] is None and not o["direction_valid"]
            for o in changed
        )
    timings = [p["processing_sec"] for p in probe.observations]
    return {
        "expected": expected,
        "passed": bool(check) and not probe.errors if expected != "report" else None,
        "capture_count": len(probe.observations),
        "detected_instances": measured,
        "match_ratio": match_ratio,
        "id_switches": switches,
        "processing_median_sec": float(np.median(timings)) if timings else None,
        "processing_max_sec": max(timings) if timings else None,
        "center_error_max_m": max((m["center_error_m"] for m in matches), default=None),
        "radius_error_max_m": max((m["radius_error_m"] for m in matches), default=None),
        "crop_contract_errors": sorted(set(probe.errors)),
        "overlay_count": len(probe.overlays),
        "marker_count": len(probe.markers),
        "statuses": sorted(set(probe.statuses)),
        "reasons": sorted({r for o in all_apples for r in o["reasons"]}),
        "counts_per_capture": [len(p["observations"]) for p in probe.observations],
        "evaluation": evaluations,
    }


def run_case(name, scene, expected, args):
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "scene.yaml"
    config.write_text(yaml.safe_dump(scene, sort_keys=False))
    rclpy.init()
    probe = Probe()
    log = (directory / "launch.log").open("w")
    command = [
        "ros2",
        "launch",
        "asac_sim",
        "topview_sim.launch.py",
        f"model_path:={args.model}",
        f"scene_file:={config.resolve()}",
        "rviz:=false",
    ]
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds and process.poll() is None:
            rclpy.spin_once(probe, timeout_sec=0.02)
        summary = summarize(probe, expected)
        summary.update(launch_command=command, launch_returncode_before_shutdown=process.poll())
        if probe.arrays and probe.arrays[0].apples:
            bridge = CvBridge()
            for apple in probe.arrays[0].apples:
                cv2.imwrite(
                    str(directory / f"{apple.apple_id}-rgb.png"),
                    bridge.imgmsg_to_cv2(apple.rgb_crop),
                )
                cv2.imwrite(
                    str(directory / f"{apple.apple_id}-mask.png"),
                    bridge.imgmsg_to_cv2(apple.mask_crop),
                )
        if probe.overlays:
            cv2.imwrite(str(directory / "overlay.png"), CvBridge().imgmsg_to_cv2(probe.overlays[0]))
        if probe.observations:
            (directory / "sample.json").write_text(
                json.dumps(probe.observations[0], indent=2, allow_nan=False)
            )
            capture = probe.observations[-1]
            s = capture["stamp"]
            try:
                tf = probe.tf.lookup_transform(
                    capture["base_frame"],
                    capture["camera_frame"],
                    Time(seconds=s["sec"], nanoseconds=s["nanosec"]),
                )
                t, q = tf.transform.translation, tf.transform.rotation
                summary["capture_tf"] = {
                    "translation": [t.x, t.y, t.z],
                    "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                }
            except Exception as error:
                summary["capture_tf_error"] = str(error)
        (directory / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(
            name,
            {
                k: v
                for k, v in summary.items()
                if k in ("passed", "capture_count", "match_ratio", "reasons", "id_switches")
            },
            flush=True,
        )
        return summary
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        log.close()
        probe.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path("src/asac_sim/config/scene.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/topview/e2e"))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--cases", default="", help="Comma-separated names; blank selects all")
    args = parser.parse_args()
    args.model = args.model.resolve()
    if not args.model.is_file():
        parser.error("Model file does not exist")
    selected = set(args.cases.split(",")) if args.cases else None
    available = scenarios(yaml.safe_load(args.scene.read_text()))
    if selected and selected - {name for name, _, _ in available}:
        parser.error("Unknown scenario name")
    results = {}
    for name, cfg, expected in available:
        if selected is None or name in selected:
            results[name] = run_case(name, cfg, expected, args)
    args.output.mkdir(parents=True, exist_ok=True)
    failed = [name for name, result in results.items() if result["passed"] is False]
    (args.output / "summary.json").write_text(
        json.dumps(
            {"tolerances_before_test": TOLERANCES, "failed_checks": failed, "cases": results},
            indent=2,
            allow_nan=False,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
