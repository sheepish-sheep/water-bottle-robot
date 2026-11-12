"""
Nav2 for the MuJoCo kitchen sim (start sim/kitchen_sim.launch.py first, mode navigation).

    ros2 launch ~/ros2_ws/src/water-bottle-robot/navigation/kitchen_nav.launch.py [controller:=rpp|dwb]

What it runs: map_server (prebuilt G-shaped Modern_1 map) + Hello Robot's
stretch_nav2 navigation stack (planner, controller, behaviors, bt_navigator),
using stretch_nav2's nav2_params.yaml with the sim overrides in sim_params().

Sim-specific choices (do NOT copy these to the real robot as-is):
  * No AMCL. The sim driver publishes a fixed identity map->odom TF and its
    /odom is the robot's true pose, so localization is already perfect. AMCL
    would fight that static TF. (Its config also listens on /scan, which the
    sim doesn't publish.) On the real robot use stretch_nav2's normal bringup.
  * use_sim_time false. The sim driver stamps scans/TF/odom with wall-clock
    time even though it publishes /clock, so everything must use wall time.
  * controller:=rpp (default) swaps stretch_nav2's DWB controller for Regulated
    Pure Pursuit. In this CPU-starved VM, DWB dithered at ~0.05 m/s next to an
    obstacle (it plans around the robot's *measured* velocity, which lags).
    controller:=dwb keeps stretch_nav2's stock DWB, for comparison.
  * controller_frequency 20 -> 10 Hz: the VM couldn't hold 20 Hz with the sim.
  * voxel_layer removed from both costmaps (duplicates obstacle_layer on a 2D
    scan; CPU starvation stalled the sim).
  * footprint: local costmap uses the robot's measured rectangle (FOOTPRINT)
    instead of robot_radius 0.22; the global (planning) costmap uses a 0.30 m
    circle so paths keep the sides/back clear. These apply to the real robot too.
  * With rpp, the final heading is NOT enforced (yaw_goal_tolerance 3.15); see
    the comments on use_rotate_to_heading below.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

# maps aren't installed by stretch_simulation, so use the source tree
MAPS = os.path.expanduser("~/ament_ws/src/stretch_ros2/stretch_simulation/maps")

# Stretch's real outline in base_link (x forward, y left), stow pose, measured
# from the MuJoCo model's collision geoms below counter height:
#   x -0.297 .. +0.095 (base_link sits near the FRONT; the base, mast and lift
#   extend ~0.30 m behind it), y -0.261 .. +0.244 (wrist on the right).
# Padded by 3 cm. stretch_nav2 instead uses robot_radius 0.22, a circle around
# base_link that misses ~0.1 m of the back end -- the rear swung into a counter
# when the robot turned.
FOOTPRINT = "[[0.125, 0.275], [0.125, -0.29], [-0.33, -0.29], [-0.33, 0.275]]"

GLOBAL_ROBOT_RADIUS = 0.30  # planning clearance for base_link (sides reach 0.29)

RPP = {
    "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
    "desired_linear_vel": 0.25,
    "lookahead_dist": 0.5,
    "min_lookahead_dist": 0.3,
    "max_lookahead_dist": 0.8,
    "lookahead_time": 1.5,
    "use_velocity_scaled_lookahead_dist": False,
    "rotate_to_heading_angular_vel": 0.8,
    # False: with True, RPP stops driving once *its* goal-distance check passes and
    # rotates in place, but the controller's goal checker measures slightly
    # differently -- the robot stalled spinning at ~0.25 m and timed out.
    "use_rotate_to_heading": False,
    "rotate_to_heading_min_angle": 0.785,
    "max_angular_accel": 3.2,
    "allow_reversing": False,
    "transform_tolerance": 0.3,
    "min_approach_linear_velocity": 0.05,
    "approach_velocity_scaling_dist": 0.6,
    "use_collision_detection": True,
    "max_allowed_time_to_collision_up_to_carrot": 1.0,
    "use_regulated_linear_velocity_scaling": True,
    "use_cost_regulated_linear_velocity_scaling": False,
    "regulated_linear_scaling_min_radius": 0.9,
    "regulated_linear_scaling_min_speed": 0.1,
    "max_robot_pose_search_dist": 10.0,
}


def set_everywhere(node, key, value):
    """Set `key` at every level of a nested dict where it already exists."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
            else:
                set_everywhere(v, key, value)


def sim_params(controller):
    """stretch_nav2's nav2_params.yaml + sim overrides, written to a temp file."""
    nav = get_package_share_directory("stretch_nav2")
    with open(os.path.join(nav, "config", "nav2_params.yaml")) as f:
        p = yaml.safe_load(f)
    set_everywhere(p, "use_sim_time", False)
    for cm in ("local_costmap", "global_costmap"):
        cp = p[cm][cm]["ros__parameters"]
        if cm == "local_costmap":
            # exact outline: RPP's collision check and the spin/backup behaviors use this
            cp.pop("robot_radius", None)
            cp["footprint"] = FOOTPRINT
            cp["footprint_padding"] = 0.0  # padding is already in FOOTPRINT
        else:
            # NavFn plans for a point and only keeps the footprint's *inscribed*
            # radius clear -- 0.125 m for the rectangle (base_link is at the front),
            # which put paths right against the box. Plan with a conservative circle.
            cp["robot_radius"] = GLOBAL_ROBOT_RADIUS
        # stretch_nav2 runs an obstacle_layer AND a 3D voxel_layer on the same 2D
        # lidar scan; the voxel layer only duplicates work. Dropping it frees CPU
        # (the sim stalled under load and Nav2 aborted / steered on stale poses).
        cp["plugins"] = [pl for pl in cp["plugins"] if pl != "voxel_layer"]
        cp.pop("voxel_layer", None)
    cs = p["controller_server"]["ros__parameters"]
    cs["controller_frequency"] = 10.0
    if controller == "rpp":
        cs["FollowPath"] = RPP
        # RPP no longer turns to the goal's heading (see use_rotate_to_heading),
        # so don't require it: the goal succeeds once within xy_goal_tolerance.
        # Facing the fountain will need a separate final rotation step.
        cs["general_goal_checker"]["yaw_goal_tolerance"] = 3.15
    elif controller != "dwb":
        raise ValueError(f"controller must be rpp or dwb, got {controller!r}")
    out = tempfile.NamedTemporaryFile("w", prefix=f"nav2_sim_{controller}_", suffix=".yaml", delete=False)
    yaml.safe_dump(p, out)
    out.close()
    return out.name


def launch_setup(context):
    params = sim_params(LC("controller").perform(context))
    nav = get_package_share_directory("stretch_nav2")
    return [
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[params, {"yaml_filename": LC("map"), "use_sim_time": False}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_map", output="screen",
             parameters=[{"use_sim_time": False, "autostart": True, "node_names": ["map_server"]}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav, "launch", "navigation_launch.py")),
            launch_arguments={"use_sim_time": "false", "autostart": "true", "params_file": params}.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=os.path.join(MAPS, "gshaped_modern1_robocasa.yaml")),
        DeclareLaunchArgument("controller", default_value="rpp", choices=["rpp", "dwb"]),
        OpaqueFunction(function=launch_setup),
    ])
