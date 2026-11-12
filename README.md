# Stretch water-bottle robot

Stretch takes a bottle from a person, drives to the water fountain, fills it
(closed-loop fill detection), and brings it back.

## Layout

| Folder | What's in it |
|---|---|
| `navigation/` | Nav2 setup. `kitchen_nav.launch.py` starts Nav2 (planner, Regulated Pure Pursuit controller, recovery behaviors) for the sim, built on Hello Robot's `stretch_nav2` config. Its docstring explains every sim-specific choice. |
| `sim/` | Simulator. `kitchen_sim.launch.py` starts the Stretch MuJoCo sim in the kitchen with a fixed spawn pose and optional extra boxes that aren't on the map. `spawn_driver.py` wraps Hello Robot's sim driver and fixes two of its bugs (see below). |
| `perception/` | `offered_bottle_detector.py`: webcam demo that finds the bottle a person is holding out (MediaPipe + YOLO). Not hooked up to ROS yet. |
| `scripts/` | Things you run. `record_nav_demo.sh` (demo with the viewer, for recordings) and `run_nav_test.sh` (headless pass/fail test). |
| `test/` | `nav_goal_test.py` sends a Nav2 goal and checks arrival; `contact_replay.py` replays the path in MuJoCo to check that no part of the robot touched anything. |

## Setup (once)

Needs Hello Robot's `stretch_ros2` in `~/ament_ws` (with `stretch_simulation`
and `stretch_nav2` built), `stretch_mujoco` installed, and the RoboCasa kitchen
assets downloaded. Then in every terminal:

```bash
source /opt/ros/humble/setup.bash && source ~/ament_ws/install/setup.bash
```

## Running

Demo with the MuJoCo viewer:

```bash
scripts/record_nav_demo.sh          # scripted: drives around an off-map box to a goal (~20 s)
scripts/record_nav_demo.sh --rviz   # also opens RViz; click "Nav2 Goal" to send your own goals
```

Also includes headless pass/fail test (exit code 0 = Nav2 reached the goal AND no robot part
came within 2 cm of the scene):
