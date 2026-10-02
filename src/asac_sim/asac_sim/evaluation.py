"""Evaluation-only matching. This module is never imported by the recognition node."""

import numpy as np
from scipy.optimize import linear_sum_assignment


TOLERANCES = {
    "sphere_center_m": 0.008,
    "sphere_radius_m": 0.005,
    "apple_center_m": 0.015,
    "apple_radius_m": 0.008,
    "match_gate_m": 0.08,
    "min_match_ratio": 0.8,
}


def compare(observations, truth, tolerances=None):
    t = dict(TOLERANCES, **(tolerances or {}))
    valid = [o for o in observations if o["center_base_valid"] and o["radius_valid"]]
    distances = np.array(
        [
            [np.linalg.norm(np.asarray(o["center_base"]) - a["center_base"]) for a in truth]
            for o in valid
        ]
    ).reshape(len(valid), len(truth))
    matches = []
    if distances.size:
        rows, cols = linear_sum_assignment(distances)
        for i, j in zip(rows, cols):
            if distances[i, j] > t["match_gate_m"]:
                continue
            o, a = valid[i], truth[j]
            radius_error = abs(o["radius_m"] - a["radius_m"])
            center_error = float(distances[i, j])
            passed = (
                center_error <= t[a["shape"] + "_center_m"]
                and radius_error <= t[a["shape"] + "_radius_m"]
            )
            matches.append(
                {
                    "truth_name": a["name"],
                    "apple_id": o.get("apple_id"),
                    "shape": a["shape"],
                    "center_error_m": center_error,
                    "radius_error_m": radius_error,
                    "within_tolerance": bool(passed),
                }
            )
    return {
        "truth_count": len(truth),
        "detected_count": len(observations),
        "metric_valid_count": len(valid),
        "matches": matches,
        "unmatched_truth_count": len(truth) - len(matches),
        "passed": bool(truth)
        and len(matches) / len(truth) >= t["min_match_ratio"]
        and all(m["within_tolerance"] for m in matches),
    }
