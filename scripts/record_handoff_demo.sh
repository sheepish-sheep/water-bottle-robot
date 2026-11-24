#!/bin/bash
# Bottle-handoff demo in MuJoCo (no ROS needed).
#
#   scripts/record_handoff_demo.sh                 # render ~/ros2_ws/artifacts/handoff/handoff_demo.mp4 (headless, ~3 min)
#   scripts/record_handoff_demo.sh --viewer        # watch it live in the MuJoCo viewer (run on the VM desktop)
#   scripts/record_handoff_demo.sh --azimuth -150  # person standing somewhere else (deg; -90 = robot's right)
#
# The robot turns to face the person, raises its lift, unfolds the gripper,
# reaches and grasps the bottle, waits for the person to let go, holds it
# upright, hands it back once the person tugs on it, and stows.
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=~/ros2_ws/artifacts/handoff
mkdir -p "$OUT"
if [ "$1" = "--viewer" ]; then
  shift
  if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "No display found. Run this from a terminal in the VM's desktop, not over SSH."; exit 1
  fi
  unset MUJOCO_GL
  exec python3 "$REPO/manipulation/handoff_sim.py" --viewer "$@"
fi
export MUJOCO_GL=egl
python3 "$REPO/manipulation/handoff_sim.py" --video "$OUT/handoff_demo.mp4" "$@" 2>&1 \
  | grep -E '"passed"|"max_tilt_deg"|"max_slip_cm"|^video:'
