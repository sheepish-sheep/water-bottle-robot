#!/bin/bash
# Nav2 demo in the kitchen sim, for screen recording.
# Run from a terminal ON THE VM DESKTOP (opens the MuJoCo viewer window).
#
#   scripts/record_nav_demo.sh            # scripted: tested start/box/goal, ~22 s of driving
#   scripts/record_nav_demo.sh --rviz     # also opens RViz; YOU click goals with "Nav2 Goal"
#
# Scripted layout (kitchen meters, same as test/nav_goal_test.py runs that PASSED):
#   start (1.3, -1.75) facing +x  ->  red box at (2.25, -1.75), not on the map
#   ->  goal (3.2, -1.75). Nav2 curves around the north side of the box.
# Override with env vars: SPAWN="x y yaw"  GOAL="x y yaw"  OBSTACLES="x,y;x,y"  CONTROLLER=rpp|dwb
#
# Ctrl+C stops everything.

RVIZ=0; [ "$1" = "--rviz" ] && RVIZ=1
read -r SX SY SYAW <<< "${SPAWN:-1.3 -1.75 0}"
read -r GX GY GYAW <<< "${GOAL:-3.2 -1.75 0}"
OBS=${OBSTACLES-2.25,-1.75}
CTRL=${CONTROLLER:-rpp}
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOGDIR=/tmp/nav_demo; mkdir -p "$LOGDIR"

source /opt/ros/humble/setup.bash
source ~/ament_ws/install/setup.bash
if [ "${HEADLESS:-0}" = 1 ]; then       # for automated checks only
  export MUJOCO_GL=egl; VIEWER=false
else
  unset MUJOCO_GL; VIEWER=true
  if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "No display found. Run this from a terminal in the VM's desktop, not over SSH."; exit 1
  fi
fi

kill_all() {
  for p in spawn_driver.py multiprocessing.spawn robot_state_publisher joint_state_publisher \
           controller_server planner_server behavior_server bt_navigator waypoint_follower \
           lifecycle_manager map_server rviz2 nav_goal_test.py; do pkill -9 -f "[${p:0:1}]${p:1}"; done
}
cleanup() {
  echo; echo "Stopping..."
  if [ -n "$REC" ]; then   # let the recorder write its trace/plot first
    kill -INT $REC 2>/dev/null
    for _ in $(seq 20); do kill -0 $REC 2>/dev/null || break; sleep 0.5; done
  fi
  kill -INT $NAV $SIM $RV 2>/dev/null; sleep 4; kill_all
  echo "Run files: $LOGDIR/run.png, run.json, run_trace.json"
}
trap cleanup EXIT
kill_nav() {
  for p in controller_server planner_server behavior_server bt_navigator waypoint_follower \
           lifecycle_manager map_server; do pkill -9 -f "[${p:0:1}]${p:1}"; done
}
# Start Nav2 and wait until bt_navigator is active. Under heavy load Nav2's
# lifecycle bringup sometimes aborts ("Failed to bring up all requested nodes"),
# so relaunch it up to 3 times.
start_nav2() {
  for nav_try in 1 2 3; do
    ros2 launch "$REPO/navigation/kitchen_nav.launch.py" controller:="${CONTROLLER:-rpp}" > "$NAVLOG" 2>&1 &
    NAV=$!; T1=$SECONDS
    while (( SECONDS - T1 < 90 )); do
      ros2 lifecycle get /bt_navigator 2>/dev/null | grep -q '^active' && return 0
      grep -q 'Aborting bringup' "$NAVLOG" && break
      sleep 2
    done
    echo "Nav2 didn't come up (try $nav_try), restarting it..."
    kill -INT $NAV 2>/dev/null; sleep 3; kill_nav; sleep 2
  done
  return 1
}

for attempt in 1 2 3; do
  echo "Starting kitchen sim (robot at $SX,$SY; obstacles: ${OBS:-none}). Takes ~30-60 s..."
  ros2 launch "$REPO/sim/kitchen_sim.launch.py" use_mujoco_viewer:=$VIEWER \
    spawn_x:="$SX" spawn_y:="$SY" spawn_yaw:="$SYAW" ${OBS:+obstacles:="$OBS"} > "$LOGDIR/sim.log" 2>&1 &
  SIM=$!; T0=$SECONDS; UP=0
  while (( SECONDS - T0 < 100 )); do
    timeout 5 ros2 topic echo --once /odom >/dev/null 2>&1 && { UP=1; break; }
    sleep 2
  done
  (( UP )) && break
  echo "Sim didn't come up (it occasionally hangs on load), retrying..."
  kill -INT $SIM 2>/dev/null; sleep 3; kill_all; sleep 2
done
(( UP )) || { echo "Sim failed to start 3 times; see $LOGDIR/sim.log"; exit 1; }

echo "Starting Nav2 ($CTRL controller)..."
NAVLOG="$LOGDIR/nav2.log"; CONTROLLER=$CTRL
start_nav2 || { echo "Nav2 failed to start 3 times; see $NAVLOG"; exit 1; }
if (( RVIZ )); then
  ros2 run rviz2 rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz > "$LOGDIR/rviz.log" 2>&1 &
  RV=$!
fi

# stow the arm + navigation mode now, so the recording starts with the robot ready
for srv in /stow_the_robot /switch_to_navigation_mode; do
  timeout 30 ros2 service call $srv std_srvs/srv/Trigger > /dev/null 2>&1
done
echo "Nav2 is ready."

if (( RVIZ )); then
  echo ">>> In RViz: click 'Nav2 Goal' (top toolbar), then click a white spot on the map and drag"
  echo "    to set the facing direction. Black = walls/counters, the colored halo = keep-out zone."
  echo "    Set as many goals as you like. Ctrl+C here when done. <<<"
  python3 "$REPO/test/nav_goal_test.py" --obstacles "$OBS" --out "$LOGDIR/run" > "$LOGDIR/recorder.log" 2>&1 &
  REC=$!
  wait $NAV
else
  echo ">>> Set up the viewer camera, start your screen recording, then press Enter. <<<"
  echo "    (Viewer: left-drag rotates, right-drag pans, scroll zooms.)"
  [ "${HEADLESS:-0}" = 1 ] || read -r
  echo "Sending goal ($GX, $GY)..."
  # Ctrl+C during the drive still saves run_trace.json / run.png
  python3 "$REPO/test/nav_goal_test.py" --goal "$GX" "$GY" "$GYAW" --obstacles "$OBS" \
    --out "$LOGDIR/run" > "$LOGDIR/goal.log" 2>&1 &
  REC=$!
  wait $REC
  grep -E '"(result|elapsed_s|final_error_m|closest_lidar_m)"|^PASS|^FAIL' "$LOGDIR/goal.log"
  REC=""
  echo "Done. Stop recording; Ctrl+C or wait to exit."
  [ "${HEADLESS:-0}" = 1 ] || sleep 5
fi
