#!/bin/bash
# Run the bottle-handoff demo in MuJoCo (no ROS needed). Simulation only.
#
#   scripts/run_handoff_demo.sh                  # tests, then the demo in the live viewer
#   scripts/run_handoff_demo.sh --azimuth -150   # person standing somewhere else (deg; -90 = robot's right)
#   scripts/run_handoff_demo.sh --random --seed 7  # a randomized person/bottle
#
# Without a desktop (e.g. over SSH) it runs the demo headless and prints pass/fail.
set -e
source "$(dirname "$0")/_env.sh"
cd "$REPO"

echo "== 1/2 planning tests =="
python3 -m pytest test/test_handoff.py -q

echo "== 2/2 handoff demo =="
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
  unset MUJOCO_GL
  "$WBR_BIN/handoff_sim" --viewer "$@"
else
  echo "(no display: running headless)"
  MUJOCO_GL=egl "$WBR_BIN/handoff_sim" "$@"
fi
