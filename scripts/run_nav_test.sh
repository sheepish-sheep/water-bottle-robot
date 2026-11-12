#!/bin/bash
# Headless end-to-end Nav2 test in the kitchen sim. Exit code 0 = PASS
# (Nav2 reached the goal AND contact_replay.py found no robot part within 2 cm of the scene).
#
#   scripts/run_nav_test.sh [GOAL_X GOAL_Y GOAL_YAW] [OBSTACLES]
#   e.g. scripts/run_nav_test.sh 1.25 -4.2 -90 "1.3,-2.4"
#
# Env: SPAWN="x y yaw" (default "1.3 -1.75 0"), CONTROLLER=rpp|dwb (default rpp).
# Spawns the robot, starts Nav2
# (kitchen_nav.launch.py), runs test/nav_goal_test.py, then shuts everything
# down. Logs + PNG/JSON go to ~/ros2_ws/artifacts/.
GX=${1:-1.25}; GY=${2:--4.2}; GYAW=${3:--90}; OBS=${4:-}
read -r SX SY SYAW <<< "${SPAWN:-1.3 -1.75 0}"
REPO=$(cd "$(dirname "$0")/.." && pwd)
ART=~/ros2_ws/artifacts; mkdir -p "$ART"
TAG="nav_${CONTROLLER:-rpp}_${GX}_${GY}${OBS:+_obs}"
source /opt/ros/humble/setup.bash
source ~/ament_ws/install/setup.bash
export MUJOCO_GL=egl

kill_sim() {
  for p in spawn_driver.py multiprocessing.spawn robot_state_publisher joint_state_publisher; do
    pkill -9 -f "[${p:0:1}]${p:1}"; done
}
cleanup() {
  kill -INT $NAV $SIM 2>/dev/null; sleep 5
  kill_sim
  for p in controller_server planner_server behavior_server bt_navigator waypoint_follower \
           lifecycle_manager map_server; do pkill -9 -f "[${p:0:1}]${p:1}"; done
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

# The sim's physics process occasionally hangs on startup (~1 in 6 launches);
# a good start takes 30-55 s, so relaunch if it isn't up after 100 s.
for attempt in 1 2 3; do
  ros2 launch "$REPO/sim/kitchen_sim.launch.py" use_mujoco_viewer:=false \
    spawn_x:="$SX" spawn_y:="$SY" spawn_yaw:="$SYAW" ${OBS:+obstacles:="$OBS"} > "$ART/${TAG}_sim.log" 2>&1 &
  SIM=$!
  T0=$SECONDS; UP=0
  while (( SECONDS - T0 < 100 )); do
    timeout 5 ros2 topic echo --once /odom >/dev/null 2>&1 && { UP=1; break; }
    sleep 2
  done
  (( UP )) && break
  echo "sim not up after 100 s (attempt $attempt), relaunching"
  kill -INT $SIM 2>/dev/null; sleep 3; kill_sim; sleep 2
done
(( UP )) || { echo "FAIL: sim never published /odom"; exit 1; }
echo "sim up after $((SECONDS - T0)) s"

NAVLOG="$ART/${TAG}_nav2.log"
start_nav2 || { echo "FAIL: Nav2 never came up"; exit 1; }
echo "Nav2 active"

timeout 300 python3 "$REPO/test/nav_goal_test.py" --goal "$GX" "$GY" "$GYAW" \
  --obstacles "$OBS" --out "$ART/$TAG" 2>&1 | grep -v -E 'Axes3D|warnings.warn'
RC=${PIPESTATUS[0]}
# real collision check: replay the path with the robot's actual body in MuJoCo
python3 "$REPO/test/contact_replay.py" "$ART/${TAG}_trace.json" --spawn "$SX" "$SY" "$SYAW" \
  --obstacles "$OBS" 2>&1 | grep -E 'cm  t=|closest_robot|CONTACT'
CRC=${PIPESTATUS[0]}
echo "nav test exit: $RC, contact check exit: $CRC   (plot: $ART/$TAG.png)"
(( RC == 0 && CRC == 0 ))
