#!/usr/bin/env python3
"""Use the shared dataset trainer on Windows/Linux without a ROS installation."""

from pathlib import Path
import sys

ASAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASAC_ROOT / 'src/asac_perception'))

from asac_perception.train import main  # noqa: E402


if __name__ == '__main__':
    main(['--workspace', str(ASAC_ROOT), *sys.argv[1:]])
