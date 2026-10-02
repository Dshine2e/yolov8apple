#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
args = parser.parse_args()
if not args.model.is_file():
    parser.error("Model file missing")
model = YOLO(str(args.model))
print(
    json.dumps(
        {
            "path": str(args.model.resolve()),
            "task": model.task,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "names": model.names,
        },
        ensure_ascii=False,
        indent=2,
    )
)
