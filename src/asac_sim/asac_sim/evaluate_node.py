"""Independent capture-stamp comparison of observations to simulator evaluation truth."""

import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from asac_sim.evaluation import compare, TOLERANCES


def stamp_key(payload):
    s = payload["stamp"]
    return (s["sec"], s["nanosec"])


class Evaluator(Node):
    def __init__(self):
        super().__init__("evaluate")
        self.duration = self.declare_parameter("duration_sec", 30.0).value
        self.output = Path(self.declare_parameter("output_file", "/tmp/asac-evaluation.json").value)
        self.tolerances = {k: self.declare_parameter(k, v).value for k, v in TOLERANCES.items()}
        self.truth, self.pending, self.results, self.previous_ids = {}, {}, [], {}
        self.id_switches = 0
        self.create_subscription(String, "/asac_sim/evaluation_truth", self._truth, 20)
        self.create_subscription(String, "/asac/observations_json", self._observation, 10)

    def _truth(self, msg):
        payload = json.loads(msg.data)
        key = stamp_key(payload)
        self.truth[key] = payload
        if key in self.pending:
            self._compare(self.pending.pop(key), payload)
        while len(self.truth) > 300:
            del self.truth[next(iter(self.truth))]

    def _observation(self, msg):
        payload = json.loads(msg.data)
        key = stamp_key(payload)
        if key in self.truth:
            self._compare(payload, self.truth[key])
        else:
            self.pending[key] = payload
            while len(self.pending) > 100:
                del self.pending[next(iter(self.pending))]

    def _compare(self, payload, truth):
        if payload["base_frame"] != truth["base_frame"]:
            self.get_logger().error("Evaluation base frame mismatch")
            return
        result = compare(payload["observations"], truth["apples"], self.tolerances)
        result["stamp"] = payload["stamp"]
        result["processing_sec"] = payload["processing_sec"]
        for match in result["matches"]:
            name, apple_id = match["truth_name"], match["apple_id"]
            if name in self.previous_ids and self.previous_ids[name] != apple_id:
                self.id_switches += 1
            self.previous_ids[name] = apple_id
        self.results.append(result)
        self.results = self.results[-1000:]

    def finish(self):
        import numpy as np

        results = self.results
        matches = [m for r in results for m in r["matches"]]
        total = sum(r["truth_count"] for r in results)
        ratio = len(matches) / total if total else 0
        times = [r["processing_sec"] for r in results]
        payload = {
            "scope": "actual_ROS_rendered_RGBD_YOLOv8_seg_capture_TF",
            "tolerances_before_test": self.tolerances,
            "compared_frames": len(results),
            "pending_unmatched_stamps": len(self.pending),
            "id_switches": self.id_switches,
            "match_ratio": ratio,
            "processing_median_sec": float(np.median(times)) if times else None,
            "processing_max_sec": max(times) if times else None,
            "passed": len(results) >= 3
            and ratio >= self.tolerances["min_match_ratio"]
            and all(m["within_tolerance"] for m in matches)
            and self.id_switches == 0,
            "frames": results,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
        self.get_logger().info(
            f"Evaluation frames={len(results)}, matched={ratio:.3f}, "
            f"passed={payload['passed']}; {self.output}"
        )
        return payload


def main(args=None):
    rclpy.init(args=args)
    node = Evaluator()
    start = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - start < node.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
