#!/usr/bin/env python3
"""
Run Hello Robot's stretch_mujoco_driver, but with a fixed robot spawn pose and
random seed in the RoboCasa kitchen (the stock driver always uses the kitchen's
default/random spawn, and objects are re-randomized each launch).

    python3 spawn_driver.py --spawn X Y YAW_DEG [--seed N] --ros-args <driver params>

X, Y are kitchen world coordinates in meters (same as /odom in the sim),
YAW_DEG is the heading (0 = +x). Normally started by kitchen_sim.launch.py.
"""
import argparse
import math
import random
import sys

import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", default="",
                    help='extra boxes not on the map, "x,y;x,y" in kitchen meters (0.3 m square, 0.8 m tall)')
    ap.add_argument("--stock-wheels", action="store_true",
                    help="keep Hello Robot's wheel actuators as-is (base moves ~1/6 of commanded speed)")
    ours, rest = ap.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest  # leave only the ROS args for rclpy.init()
    return ours


def add_obstacles(xml, spec):
    """Insert 0.3 x 0.3 x 0.8 m red boxes into the scene's worldbody. Returns (xml, count)."""
    boxes = []
    for i, item in enumerate(filter(None, spec.replace(" ", "").split(";"))):
        x, y = (float(v) for v in item.split(","))
        boxes.append(f'<body name="extra_obstacle_{i}" pos="{x} {y} 0.4">'
                     f'<geom type="box" size="0.15 0.15 0.4" rgba="0.9 0.1 0.1 1"/></body>')
    if boxes:
        xml = xml.replace("<worldbody>", "<worldbody>" + "".join(boxes), 1)
    return xml, len(boxes)


WHEEL_KV = 180.0  # same torque-per-(rad/s)-error as the stock gear=3, kv=20 servo


def fix_wheels(model):
    """
    The stock wheel servos have gear=3, but stretch_mujoco already sends wheel
    speeds in rad/s (diff_drive_inv_kinematics), so the wheel settles at 1/3 of
    the command, and joint friction (frictionloss=35) eats more: 0.4 m/s
    commanded gives ~0.06 m/s. With gear=1 the wheel tracks the command. The
    ctrlrange of +-6 rad/s is kept, so the base still tops out at ~0.30 m/s,
    in line with the real robot.
    """
    for name in ("left_wheel_vel", "right_wheel_vel"):
        a = model.actuator(name)
        a.gear[0] = 1.0
        a.gainprm[0] = WHEEL_KV
        a.biasprm[2] = -WHEEL_KV
    print(f"[spawn_driver] wheel servos: gear 3 -> 1, kv -> {WHEEL_KV} (use --stock-wheels to disable)", flush=True)


def fix_zero_move_by():
    """
    When cmd_vel stops, the driver "holds" the base with move_by(base_translate, 0.0),
    but stretch_mujoco's _base_translate_by uses sign = -1 for 0 and drives the
    base BACKWARDS at 0.3 m/s for a tick, on every driver loop for ~1 s.
    Treat a zero base move as a plain stop instead.
    """
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.actuators import Actuators

    original = StretchMujocoSimulator.move_by

    def move_by(self, actuator, pos, *a, **kw):
        if pos == 0.0 and actuator in (Actuators.base_translate, Actuators.base_rotate):
            return self.set_base_velocity(0.0, 0.0)
        return original(self, actuator, pos, *a, **kw)

    StretchMujocoSimulator.move_by = move_by


def main():
    args = parse()
    fix_zero_move_by()
    random.seed(args.seed)
    np.random.seed(args.seed)

    x, y, yaw_deg = args.spawn
    half = math.radians(yaw_deg) / 2
    # MJCF body quat order is "w x y z"
    pose = {"pos": f"{x} {y} 0", "quat": f"{math.cos(half)} 0 0 {math.sin(half)}"}

    from stretch_mujoco_driver import stretch_mujoco_driver as drv

    original = drv.model_generation_wizard

    def with_spawn(*a, **kw):
        kw["robot_spawn_pose"] = pose
        model, xml, info = original(*a, **kw)
        if not args.stock_wheels:
            fix_wheels(model)
        return model, xml, info

    drv.model_generation_wizard = with_spawn

    if args.obstacles:
        # insert the boxes in the generator's own XML step, so the kitchen is
        # still compiled only once (a second full compile made startup hang)
        from stretch_mujoco import robocasa_gen
        add_stretch = robocasa_gen.add_stretch_to_kitchen

        def add_stretch_and_obstacles(xml, pose_attrib):
            xml, n = add_obstacles(xml, args.obstacles)
            print(f"[spawn_driver] added {n} obstacle box(es): {args.obstacles}", flush=True)
            return add_stretch(xml, pose_attrib)

        robocasa_gen.add_stretch_to_kitchen = add_stretch_and_obstacles
    print(f"[spawn_driver] spawning Stretch at x={x} y={y} yaw={yaw_deg} deg, seed={args.seed}", flush=True)
    drv.main()


if __name__ == "__main__":  # required: the sim starts its physics process with multiprocessing 'spawn'
    main()
