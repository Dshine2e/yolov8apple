"""Session IDs with gated association and finite retention; no guaranteed re-ID."""

from dataclasses import dataclass
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    apple_id: str
    bbox: list
    center: object
    stamp: float


class Tracker:
    def __init__(
        self, max_distance_m=0.06, min_iou=0.1, ttl_sec=2.0, duplicate_iou=0.85, session=None
    ):
        self.session = session or uuid.uuid4().hex[:12]
        self.counter = 0
        self.tracks = []
        self.max_distance, self.min_iou, self.ttl = max_distance_m, min_iou, ttl_sec
        self.duplicate_iou = duplicate_iou
        self.last_stamp = None
        self.frame = None

    def update(self, observations, stamp, frame):
        if self.frame != frame or (self.last_stamp is not None and stamp < self.last_stamp):
            self.tracks = []  # counter stays monotonic after bag rewind/frame changes
        self.frame, self.last_stamp = frame, stamp
        self.tracks = [t for t in self.tracks if stamp - t.stamp <= self.ttl]
        unique = []
        for o in sorted(observations, key=lambda o: o["confidence"], reverse=True):
            if not any(bbox_iou(o["bbox"], p["bbox"]) >= self.duplicate_iou for p in unique):
                unique.append(o)
        costs = np.full((len(self.tracks), len(unique)), 1e6)
        for i, t in enumerate(self.tracks):
            for j, o in enumerate(unique):
                iou = bbox_iou(t.bbox, o["bbox"])
                center = o.get("center_camera")
                if center is not None and t.center is not None:
                    distance = np.linalg.norm(np.asarray(center) - t.center)
                    if distance <= self.max_distance:
                        costs[i, j] = 0.7 * distance / self.max_distance + 0.3 * (1 - iou)
                elif iou >= self.min_iou:
                    costs[i, j] = 1 - iou
        matches = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            matches = {j: i for i, j in zip(rows, cols) if costs[i, j] < 1e5}
        for j, o in enumerate(unique):
            if j in matches:
                t = self.tracks[matches[j]]
                o["tracking_method"] = (
                    "3d_distance_and_2d_iou"
                    if t.center is not None and o.get("center_camera") is not None
                    else "2d_iou_fallback"
                )
                t.bbox, t.center, t.stamp = o["bbox"], o.get("center_camera"), stamp
            else:
                self.counter += 1
                t = Track(
                    f"{self.session}-{self.counter:06d}", o["bbox"], o.get("center_camera"), stamp
                )
                self.tracks.append(t)
                o["tracking_method"] = "new_track"
            o["apple_id"] = t.apple_id
        return unique
