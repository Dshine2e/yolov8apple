#!/usr/bin/env bash
# Download the previously selected export, train, evaluate, then select best.pt.
set -eo pipefail
ASAC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ASAC_ROOT"
if [[ ! -x .venv/bin/python || ! -x .venv-roboflow/bin/python ]]; then
  echo 'Install .venv and .venv-roboflow first; see docs/ROBOFLOW_TRAINING.md' >&2
  exit 1
fi
if [[ ! -f datasets/apple-v2/data.yaml ]]; then
  if [[ -f src/asac_perception/resource/Apple.v2i.yolov8.zip ]]; then
    .venv-roboflow/bin/python scripts/download_dataset.py \
      --zip src/asac_perception/resource/Apple.v2i.yolov8.zip --output datasets/apple-v2
  else
    .venv-roboflow/bin/python scripts/download_dataset.py --output datasets/apple-v2
  fi
fi
export PYTHONPATH="$ASAC_ROOT/src/asac_perception${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m asac_perception.train \
  --data datasets/apple-v2/data.yaml --task detect --model models/yolov8n.pt \
  --epochs 100 --imgsz 640 --batch 8 --device cpu --workers 2 \
  --project runs/apple --name yolov8n_roboflow_v2 --workspace "$ASAC_ROOT" --apply "$@"
