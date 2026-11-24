# Sourced by the other scripts: ROS, Hello Robot's workspace, and the workspace
# this repo is built in (default: two folders up, i.e. ~/ros2_ws for
# ~/ros2_ws/src/water-bottle-robot; override with WBR_WS=/path/to/ws).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WS=${WBR_WS:-$(cd "$REPO/../.." && pwd)}
source /opt/ros/humble/setup.bash
source ~/ament_ws/install/setup.bash
if [ -f "$WS/install/setup.bash" ]; then source "$WS/install/setup.bash"; fi
if ! WBR_PREFIX=$(ros2 pkg prefix water_bottle_robot 2>/dev/null); then
  echo "water_bottle_robot isn't built. Build it first:"
  echo "  cd $WS && colcon build --symlink-install --packages-select water_bottle_robot"
  exit 1
fi
WBR_BIN="$WBR_PREFIX/lib/water_bottle_robot"   # installed commands (same as 'ros2 run water_bottle_robot ...')
