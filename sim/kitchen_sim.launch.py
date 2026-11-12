"""
Stretch MuJoCo sim in the RoboCasa kitchen with a FIXED robot spawn pose.
Same nodes as stretch_simulation/stretch_mujoco_driver.launch.py, but the driver
is started through spawn_driver.py so the start pose and seed are repeatable.

    ros2 launch ~/ros2_ws/src/water-bottle-robot/sim/kitchen_sim.launch.py \
        spawn_x:=3.5 spawn_y:=-1.2 spawn_yaw:=0 use_mujoco_viewer:=true
"""
from pathlib import Path

import launch_ros.parameter_descriptions
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration as LC
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent

ARGS = {
    # Defaults: facing the east counter, lidar ~0.3 m from it. Tested (fast
    # wheels): ~5 s setup, ~6 s turning in place, then drives across the
    # kitchen and out through the gap -- fits a ~20 s recording.
    "spawn_x": "3.5",
    "spawn_y": "-1.2",
    "spawn_yaw": "0",          # degrees, 0 = +x
    "seed": "0",               # fixes the counter-object placement
    "obstacles": "",           # extra boxes not on the map: "x,y;x,y" (kitchen meters)
    "layout": "G-shaped",
    "style": "Modern_1",
    "use_mujoco_viewer": "true",
    "use_cameras": "false",
    "mode": "navigation",
}


def generate_launch_description():
    ld = LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()])

    desc = get_package_share_path("stretch_description") / "urdf"
    urdf = desc / "stretch.urdf"
    if not urdf.is_file():
        urdf = desc / "stretch_description_SE3_eoa_wrist_dw3_tool_sg3.xacro"
    robot_description = launch_ros.parameter_descriptions.ParameterValue(
        Command(["xacro ", str(urdf)]), value_type=str)

    ld.add_action(Node(
        package="joint_state_publisher", executable="joint_state_publisher", output="log",
        parameters=[{"source_list": ["/stretch/joint_states"]}, {"rate": 30.0},
                    {"robot_description": robot_description}],
        arguments=["--ros-args", "--log-level", "error"]))
    ld.add_action(Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="both",
        parameters=[{"robot_description": robot_description}, {"publish_frequency": 30.0}],
        arguments=["--ros-args", "--log-level", "error"]))

    ld.add_action(ExecuteProcess(
        cmd=["python3", str(HERE / "spawn_driver.py"),
             "--spawn", LC("spawn_x"), LC("spawn_y"), LC("spawn_yaw"), "--seed", LC("seed"), ["--obstacles=", LC("obstacles")],
             "--ros-args",
             "-r", "__node:=stretch_mujoco_driver",
             "-r", "cmd_vel:=/stretch/cmd_vel",
             "-r", "joint_states:=/stretch/joint_states",
             "-p", "rate:=30.0", "-p", "timeout:=0.5",
             "-p", "broadcast_odom_tf:=true", "-p", "fail_out_of_range_goal:=false",
             "-p", ["mode:=", LC("mode")],
             "-p", ["use_mujoco_viewer:=", LC("use_mujoco_viewer")],
             "-p", ["use_cameras:=", LC("use_cameras")],
             "-p", "use_robocasa:=true",
             "-p", "robocasa_task:=PnPCounterToCab",
             "-p", ["robocasa_layout:=", LC("layout")],
             "-p", ["robocasa_style:=", LC("style")]],
        output="screen", emulate_tty=True))
    return ld
