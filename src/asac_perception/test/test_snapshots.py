import json

import numpy as np

from asac_perception.snapshots import SnapshotWriter


def test_snapshot_interval_quota_and_null_metadata(tmp_path):
    writer = SnapshotWriter(str(tmp_path), min_interval_sec=10, max_batches=2)
    observation = {
        "apple_id": "session-000001",
        "reasons": ["rgb_only_no_depth"],
        "rgb_crop": np.zeros((8, 8, 3), np.uint8),
        "mask_crop": np.ones((8, 8), np.uint8) * 255,
        "full_mask": np.ones((8, 8), bool),
        "center_base": None,
        "radius_m": None,
    }
    assert writer.write([observation], {"stamp": {"sec": 1}}, 1)
    for stamp in range(2, 10):
        assert not writer.write([observation], {}, stamp)
    writer.requested = True
    assert writer.write([observation], {"stamp": {"sec": 10}}, 10)
    writer.requested = True
    assert not writer.write([observation], {}, 100)
    batches = list(writer.directory.iterdir())
    assert len(batches) == 2
    for batch in batches:
        payload = json.loads((batch / "observations.json").read_text())
        assert payload["observations"][0]["center_base"] is None
        assert payload["observations"][0]["radius_m"] is None
        assert len(list(batch.glob("*.png"))) == 2
