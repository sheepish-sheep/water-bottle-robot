# Stretch water-bottle robot

Stretch takes a bottle from a person, drives to the water fountain, fills it
(closed-loop fill detection), and brings it back.

## Layout

A ROS 2 (ament_python) package, `water_bottle_robot`.

| Folder | What's in it |
|---|---|
| `water_bottle_robot/navigation/` | `go_to_fountain.py`: drive to the nearest water fountain (Nav2 plans, reactive or Nav2 driving). |
| `water_bottle_robot/reactive/` | `obstacle_avoidance.py`: the reactive obstacle-avoidance decision used by `go_to_fountain`. |
| `water_bottle_robot/manipulation/` | Bottle handoff: Stretch takes a bottle a person holds out, holds it upright, and hands it back. MuJoCo sim with randomized checks (`handoff_sim`), and real-robot versions on `stretch_body` (`handoff_real`) and on Hello Robot's ROS driver (`handoff_ros`). See `docs/handoff.md` and `docs/handoff_real_robot.md`. |
| `water_bottle_robot/perception/` | `gripper_bottle_detector.py`: ROS node that finds the bottle in the gripper camera (YOLO) and publishes where it is; `handoff_ros --camera` uses it to line up before grasping ([docs/gripper_camera.md](docs/gripper_camera.md)). `offered_bottle_detector.py`: webcam demo that finds the bottle a person is holding out (MediaPipe + YOLO). |
| `water_bottle_robot/sim/` | `spawn_driver.py` wraps Hello Robot's MuJoCo driver: fixed spawn pose, off-map boxes, and fixes for two of its bugs (see below). |
| `water_bottle_robot/tools/` | Sim test tools: `nav_goal_test` sends a Nav2 goal and checks arrival; `contact_replay` replays the path in MuJoCo to check that no part of the robot touched anything. |
| `launch/` | `kitchen_sim.launch.py` (the MuJoCo kitchen with a fixed spawn) and `kitchen_nav.launch.py` (Nav2 for it: planner, Regulated Pure Pursuit controller, recovery behaviors, built on Hello Robot's `stretch_nav2`; its docstring explains every sim-specific choice). |
| `scripts/` | Things you run. `record_nav_demo.sh` (demo with the viewer, for recordings), `run_nav_test.sh` (headless Nav2 pass/fail test), `run_handoff_demo.sh` (bottle-handoff demo in MuJoCo), `run_handoff_ros_sim.sh` (ROS handoff dry run against the MuJoCo driver). |
| `test/` | pytest tests (`colcon test`). |
| `docs/` | Handoff write-up and the real-robot procedure. |

Commands (`ros2 run water_bottle_robot <name>`): `handoff_sim`, `handoff_real`,
`handoff_real_check`, `handoff_ros`, `go_to_fountain`, `obstacle_avoidance`,
`offered_bottle_detector`, `gripper_bottle_detector`, `nav_goal_test`, `contact_replay`.

## Setup (once)

Needs Hello Robot's `stretch_ros2` in `~/ament_ws` (with `stretch_simulation`
and `stretch_nav2` built), `stretch_mujoco` installed, and the RoboCasa kitchen
assets downloaded. Clone this repo into a colcon workspace's `src/` (on the VM:
`~/ros2_ws/src`; on the robot: `~/ament_ws/src`) and build it:

```bash
source /opt/ros/humble/setup.bash && source ~/ament_ws/install/setup.bash
cd ~/ros2_ws && colcon build --symlink-install --packages-select water_bottle_robot
```

Then in every terminal:

```bash
source /opt/ros/humble/setup.bash && source ~/ament_ws/install/setup.bash && source ~/ros2_ws/install/setup.bash
```

With `--symlink-install`, edits to the Python files take effect without
rebuilding; new files, launch files or commands need a rebuild. Tests:
`colcon test --packages-select water_bottle_robot && colcon test-result --verbose`
(or `python3 -m pytest test` in the repo).

## Running

Demo with the MuJoCo viewer:

```bash
scripts/record_nav_demo.sh          # scripted: drives around an off-map box to a goal (~20 s)
scripts/record_nav_demo.sh --rviz   # also opens RViz; click "Nav2 Goal" to send your own goals
```

Also includes headless pass/fail test (exit code 0 = Nav2 reached the goal AND no robot part
came within 2 cm of the scene):
