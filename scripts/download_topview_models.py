#!/usr/bin/env python3
"""Fetch pinned official checkpoints with SHA256 validation, never overwrite existing files."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


HASHES = {
    "yolov8n-seg.pt": "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c",
    "yolov8s-seg.pt": "0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413",
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
}
BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=HASHES, default=["yolov8n-seg.pt"])
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        path = args.directory / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
                raise ValueError(
                    f"Existing checkpoint differs; preserved without overwrite: {path}"
                )
            print(f"Already verified: {path}")
            continue
        temporary = path.with_suffix(".pt.part")
        with urlopen(BASE + name, timeout=60) as response, temporary.open("xb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != HASHES[name]:
            raise ValueError(f"SHA256 mismatch; partial file preserved: {temporary}")
        temporary.rename(path)
        print(f"Downloaded and verified: {path}")


if __name__ == "__main__":
    main()
