#!/usr/bin/env bash
set -eo pipefail
ASAC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASAC_MODE="${1:-sim}"
ASAC_MODEL="${2:-$ASAC_ROOT/models/yolov8n-seg.pt}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
ASAC_SELECTED_DISTRO="${ASAC_ROS_DISTRO:-${ROS_DISTRO:-humble}}"
if [[ "$ASAC_MODE" != sim && "$ASAC_MODE" != real ]]; then
  echo 'Usage: run_topview.sh sim|real [segmentation.pt] [launch arguments]' >&2
  exit 1
fi
for ASAC_REQUIRED in "/opt/ros/$ASAC_SELECTED_DISTRO/setup.bash" "$ASAC_ROOT/.venv/bin/activate" "$ASAC_ROOT/install/local_setup.bash" "$ASAC_MODEL"; do
  if [[ ! -f "$ASAC_REQUIRED" ]]; then
    echo "Required file missing: $ASAC_REQUIRED; see docs/TOPVIEW.md" >&2
    exit 1
  fi
done
source "/opt/ros/$ASAC_SELECTED_DISTRO/setup.bash"
source "$ASAC_ROOT/.venv/bin/activate"
source "$ASAC_ROOT/install/local_setup.bash"
if [[ "$ASAC_MODE" == sim ]]; then
  exec ros2 launch asac_sim topview_sim.launch.py model_path:="$ASAC_MODEL" "$@"
else
  exec ros2 launch asac_perception topview_real.launch.py model_path:="$ASAC_MODEL" "$@"
fi
