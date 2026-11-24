#!/bin/bash
# The ROS handoff (handoff_ros) against Hello Robot's MuJoCo driver, headless.
# Exit code 0 = PASS. Starts the sim (our launch, empty scene, position mode), runs the
# dry run (reach to the bottle spot with the gripper open, come back, stow),
# then shuts everything down. Only the dry run is meaningful in this sim: no
# person or bottle, and the sim driver reports zero arm effort.
#
#   scripts/run_handoff_ros_sim.sh [handoff_ros options]   # default: --distance 0.70 --hand-top 0.83
source "$(dirname "$0")/_env.sh"
ART=~/ros2_ws/artifacts/handoff_ros; mkdir -p "$ART"
export MUJOCO_GL=egl
export WBR_LOG_DIR=$ART
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(--distance 0.70 --hand-top 0.83)

kill_sim() {
  for p in spawn_driver multiprocessing.spawn robot_state_publisher joint_state_publisher; do
    pkill -9 -f "[${p:0:1}]${p:1}"; done
}
trap 'kill -INT $SIM 2>/dev/null; sleep 3; kill_sim' EXIT

for attempt in 1 2 3; do     # the sim's startup occasionally hangs: relaunch after 100 s
  # our launch: its spawn_driver.py fixes a sim-driver bug that aborts every arm goal
  # (empty scene: the arm reaches into open space)
  ros2 launch water_bottle_robot kitchen_sim.launch.py mode:=position use_mujoco_viewer:=false \
    use_robocasa:=false > "$ART/sim.log" 2>&1 &
  SIM=$!; T0=$SECONDS; UP=0
  while (( SECONDS - T0 < 100 )); do
    timeout 5 ros2 topic echo --once /stretch/joint_states >/dev/null 2>&1 && { UP=1; break; }
    sleep 2
  done
  (( UP )) && break
  echo "sim not up after 100 s (attempt $attempt), relaunching"
  kill -INT $SIM 2>/dev/null; sleep 3; kill_sim; sleep 2
done
(( UP )) || { echo "FAIL: sim never published /stretch/joint_states"; exit 1; }
echo "sim up after $((SECONDS - T0)) s"

timeout 400 "$WBR_BIN/handoff_ros" "${ARGS[@]}" --dry-run --yes 2>&1 | tee "$ART/handoff_ros.log" \
  | grep -vE "move_to_pose|Waiting on|connected to"
RC=${PIPESTATUS[0]}
echo "handoff_ros exit: $RC"
exit $RC
