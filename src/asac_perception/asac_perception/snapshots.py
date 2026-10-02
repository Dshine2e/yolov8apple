"""Bounded RGB/mask/JSON snapshots: state changes, periodic samples, or service request."""

import json
from pathlib import Path
import time

import cv2

from asac_perception.observations import metadata


class SnapshotWriter:
    def __init__(self, directory="", min_interval_sec=10.0, max_batches=50):
        self.directory = Path(directory).expanduser() if directory else None
        self.interval, self.limit = min_interval_sec, max_batches
        self.count, self.last = 0, float("-inf")
        self.last_state = None
        self.requested = False
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Unique run directory preserves earlier runs; quota is explicitly per run.
            self.directory = self.directory / f"run-{time.time_ns()}"
            self.directory.mkdir()

    def write(self, observations, header, stamp):
        state = tuple(sorted((o["apple_id"], tuple(o["reasons"])) for o in observations))
        changed = state != self.last_state
        if self.directory is None or self.count >= self.limit:
            return False
        if stamp - self.last < self.interval and not self.requested:
            return False
        if not observations and not changed and not self.requested:
            return False
        batch = self.directory / f"{self.count:04d}-{int(stamp * 1e9)}"
        batch.mkdir()
        for o in observations:
            cv2.imwrite(str(batch / f"{o['apple_id']}-rgb.png"), o["rgb_crop"])
            cv2.imwrite(str(batch / f"{o['apple_id']}-mask.png"), o["mask_crop"])
        payload = dict(header, observations=[metadata(o) for o in observations])
        (batch / "observations.json").write_text(json.dumps(payload, allow_nan=False, indent=2))
        self.count += 1
        self.last, self.last_state, self.requested = stamp, state, False
        return True
