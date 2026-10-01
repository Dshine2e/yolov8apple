#!/usr/bin/env bash
# 첫 인자: 모델 경로(선택), 이후 인자: 추가 ROS launch 인자
set -eo pipefail
ASAC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASAC_SELECTED_DISTRO="${ASAC_ROS_DISTRO:-${ROS_DISTRO:-humble}}"
ASAC_MODEL="${1:-$ASAC_ROOT/models/yolov8n.pt}"
if (( $# > 0 )); then
  shift
fi
if [[ ! -f "/opt/ros/$ASAC_SELECTED_DISTRO/setup.bash" ]]; then
  echo "ROS setup not found: /opt/ros/$ASAC_SELECTED_DISTRO/setup.bash" >&2
  exit 1
fi
if [[ ! -f "$ASAC_ROOT/.venv/bin/activate" || ! -f "$ASAC_ROOT/install/local_setup.bash" ]]; then
  echo "Create the project venv and build the workspace as described in README.md" >&2
  exit 1
fi
if [[ ! -f "$ASAC_MODEL" ]]; then
  echo "Model weights not found: $ASAC_MODEL; supply an existing .pt path" >&2
  exit 1
fi
source "/opt/ros/$ASAC_SELECTED_DISTRO/setup.bash"
source "$ASAC_ROOT/.venv/bin/activate"
source "$ASAC_ROOT/install/local_setup.bash"
exec ros2 launch asac_perception d455_yolo.launch.py \
  model_path:="$ASAC_MODEL" device:="${ASAC_DEVICE:-cpu}" \
  target_classes:="${ASAC_TARGET_CLASSES:-apple}" "$@"
