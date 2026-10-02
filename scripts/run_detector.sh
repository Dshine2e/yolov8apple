#!/usr/bin/env bash
# 첫 인자: 모델 경로(선택), 이후 인자: 추가 ROS launch 인자
set -eo pipefail
ASAC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASAC_SELECTED_DISTRO="${ASAC_ROS_DISTRO:-${ROS_DISTRO:-humble}}"
ASAC_DEFAULT_CLASSES=apple
if (( $# > 0 )) && [[ "$1" != *:=* ]]; then
  ASAC_MODEL="$1"
  shift
else
  if [[ ! -x "$ASAC_ROOT/.venv/bin/python" ]]; then
    echo 'Create the project venv as described in README.md' >&2
    exit 1
  fi
  ASAC_SELECTION_OUTPUT="$("$ASAC_ROOT/.venv/bin/python" \
    "$ASAC_ROOT/src/asac_perception/asac_perception/model_selection.py" --root "$ASAC_ROOT")"
  mapfile -t ASAC_SELECTION <<< "$ASAC_SELECTION_OUTPUT"
  ASAC_MODEL="${ASAC_SELECTION[0]}"
  ASAC_DEFAULT_CLASSES="${ASAC_SELECTION[1]-}"
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
ASAC_CLASS_ARGUMENTS=()
ASAC_CLASS_FILTER="${ASAC_TARGET_CLASSES-$ASAC_DEFAULT_CLASSES}"
if [[ -n "$ASAC_CLASS_FILTER" ]]; then
  ASAC_CLASS_ARGUMENTS=("target_classes:=$ASAC_CLASS_FILTER")
fi
exec ros2 launch asac_perception d455_yolo.launch.py \
  model_path:="$ASAC_MODEL" device:="${ASAC_DEVICE:-cpu}" \
  "${ASAC_CLASS_ARGUMENTS[@]}" "$@"
