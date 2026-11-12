#!/usr/bin/env python3
"""
Pass/fail Nav2 goal test against the running kitchen sim + kitchen_nav.launch.py.

    python3 test/nav_goal_test.py --goal 1.25 -4.2 -90 [--timeout 120] [--out PREFIX]
          [--obstacles "x,y;x,y"]   # only for drawing; spawn them via sim/kitchen_sim.launch.py

Without --goal it only records (path, plan, costmap) until Ctrl+C -- use that
while sending goals by hand from RViz. Ctrl+C always still writes the files.

Stows the arm, switches to navigation mode, sends one NavigateToPose goal, and
records /odom and /scan_filtered while it drives. PASS if Nav2 reports success
within the timeout, the robot ends within --tol of the goal, and no lidar
return ever came closer than --min-clearance (a collision proxy: the base's
half-width is ~0.17 m). Writes PREFIX.json and PREFIX.png; exit code 0 = PASS.
Only talks to the SIM: it refuses to run unless /clock is being published.
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                    help="omit to just record until Ctrl+C")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--nav2-timeout", type=float, default=90.0, help="s to wait for Nav2 to come up")
    ap.add_argument("--tol", type=float, default=0.35, help="m, max final distance to goal")
    ap.add_argument("--min-clearance", type=float, default=0.20, help="m, closest allowed lidar return")
    ap.add_argument("--obstacles", default="", help='"x,y;x,y", drawn on the plot')
    ap.add_argument("--out", default=os.path.expanduser("~/ros2_ws/artifacts/nav_goal"))
    ap.add_argument("--map", default=os.path.expanduser(
        "~/ament_ws/src/stretch_ros2/stretch_simulation/maps/gshaped_modern1_robocasa.yaml"))
    args = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Trigger

    from rclpy.signals import SignalHandlerOptions
    # handle Ctrl+C ourselves so the trace/plot still get written
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    interrupted = []
    phase = ["setup"]

    def on_signal(*_):
        if phase[0] == "setup":       # nothing recorded yet: just quit
            raise SystemExit(130)
        interrupted.append(1)         # driving/recording: stop and still write files
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    nav = BasicNavigator()
    poses, mins, clock_seen = [], [], []
    nav.create_subscription(Odometry, "/odom", lambda m: poses.append(
        (time.monotonic(), m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))), 10)

    def on_scan(m):
        r = np.asarray(m.ranges, float)
        ok = np.isfinite(r) & (r > m.range_min) & (r < m.range_max)
        if ok.any():
            mins.append(float(r[ok].min()))
    nav.create_subscription(LaserScan, "/scan_filtered", on_scan, qos_profile_sensor_data)
    nav.create_subscription(Clock, "/clock", lambda m: clock_seen.append(1), 10)
    from geometry_msgs.msg import Twist
    cmds = []  # (t, v, w) commanded by Nav2, for debugging controller behavior
    nav.create_subscription(Twist, "/stretch/cmd_vel",
                            lambda m: cmds.append((time.monotonic(), m.linear.x, m.angular.z)), 10)
    latest = {}  # last global plan + global costmap, for the plot
    nav.create_subscription(Path, "/plan", lambda m: latest.__setitem__("plan", m), 10)
    nav.create_subscription(OccupancyGrid, "/global_costmap/costmap", lambda m: latest.__setitem__("costmap", m),
                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    # safety: only ever drive the simulator
    t0 = time.monotonic()
    while not clock_seen and time.monotonic() - t0 < 10:
        rclpy.spin_once(nav, timeout_sec=0.2)
    if not clock_seen:
        print("FAIL: no /clock -- this test only runs against the MuJoCo sim")
        sys.exit(2)

    for srv in ("/stow_the_robot", "/switch_to_navigation_mode"):
        cli = nav.create_client(Trigger, srv)
        if cli.wait_for_service(timeout_sec=5.0):
            fut = cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(nav, fut, timeout_sec=20.0)
            print(f"{srv}: {fut.result().message if fut.result() else 'timeout'}")

    # wait for Nav2 (no AMCL in sim, see kitchen_nav.launch.py), but not forever:
    # under heavy load its lifecycle bringup can abort and never come up
    from lifecycle_msgs.srv import GetState
    cli = nav.create_client(GetState, "/bt_navigator/get_state")
    deadline = time.monotonic() + args.nav2_timeout
    while True:
        if cli.wait_for_service(timeout_sec=1.0):
            fut = cli.call_async(GetState.Request())
            rclpy.spin_until_future_complete(nav, fut, timeout_sec=2.0)
            if fut.result() is not None and fut.result().current_state.label == "active":
                break
        if time.monotonic() > deadline:
            print(f"FAIL: Nav2 (bt_navigator) not active after {args.nav2_timeout:.0f} s")
            sys.exit(3)
        time.sleep(1.0)
    print("Nav2 is active", flush=True)
    phase[0] = "run"

    poses.clear(); mins.clear(); cmds.clear()
    start = time.monotonic()
    timed_out = False
    if args.goal:
        gx, gy, gyaw = args.goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = nav.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = gx, gy
        goal.pose.orientation.z = math.sin(math.radians(gyaw) / 2)
        goal.pose.orientation.w = math.cos(math.radians(gyaw) / 2)
        nav.goToPose(goal)
        while not nav.isTaskComplete() and not interrupted:
            rclpy.spin_once(nav, timeout_sec=0.1)
            if time.monotonic() - start > args.timeout:
                timed_out = True
                break
        if timed_out or interrupted:
            nav.cancelTask()
        result = nav.getResult()
    else:
        print("recording until Ctrl+C ...", flush=True)
        while not interrupted:
            rclpy.spin_once(nav, timeout_sec=0.1)
        gx = gy = float("nan")
        result = None
    elapsed = time.monotonic() - start
    for _ in range(5):
        rclpy.spin_once(nav, timeout_sec=0.1)

    P = np.array([p[1:] for p in poses]) if poses else np.zeros((0, 3))
    final_err = float(math.hypot(P[-1, 0] - gx, P[-1, 1] - gy)) if len(P) and args.goal else float("inf")
    path_len = float(np.sum(np.hypot(*np.diff(P[:, :2], axis=0).T))) if len(P) > 1 else 0.0
    closest = min(mins) if mins else float("nan")
    checks = {
        "nav2_succeeded": (result == TaskResult.SUCCEEDED) and not timed_out,
        "within_tolerance": final_err <= args.tol,
        "no_near_collision": bool(mins) and closest >= args.min_clearance,
    }
    summary = dict(goal=args.goal, result="INTERRUPTED" if interrupted else str(result),
                   timed_out=timed_out, elapsed_s=round(elapsed, 1),
                   path_length_m=round(path_len, 2), final_error_m=round(final_err, 3),
                   closest_lidar_m=round(closest, 3), checks=checks, passed=all(checks.values()),
                   start=P[0].round(2).tolist() if len(P) else None, end=P[-1].round(2).tolist() if len(P) else None)
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out + "_trace.json", "w") as f:  # raw time series for debugging
        json.dump({"t0": start, "odom": poses, "cmd_vel": cmds}, f)

    # plot path over the map
    import cv2
    import matplotlib
    import yaml
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = yaml.safe_load(open(args.map))
    img = cv2.imread(os.path.join(os.path.dirname(args.map), meta["image"]), -1)
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(img, cmap="gray", extent=[ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res])
    if "costmap" in latest:  # Nav2's global costmap: lethal/inscribed cells in orange
        cm = latest["costmap"]; i = cm.info
        c = np.array(cm.data, dtype=float).reshape(i.height, i.width)
        c = np.ma.masked_where(c < 99, c)
        ax.imshow(c, origin="lower", cmap="autumn", alpha=0.6, extent=[
            i.origin.position.x, i.origin.position.x + i.width * i.resolution,
            i.origin.position.y, i.origin.position.y + i.height * i.resolution])
    if "plan" in latest and latest["plan"].poses:
        q = np.array([[p.pose.position.x, p.pose.position.y] for p in latest["plan"].poses])
        ax.plot(q[:, 0], q[:, 1], "--", c="tab:purple", lw=1.5, label="last global plan")
    for item in filter(None, args.obstacles.replace(" ", "").split(";")):
        x, y = (float(v) for v in item.split(","))
        ax.add_patch(plt.Rectangle((x - 0.15, y - 0.15), 0.3, 0.3, color="tab:red", label="off-map obstacle"))
    if len(P):
        ax.plot(P[:, 0], P[:, 1], c="tab:blue", lw=2, label="robot path")
        ax.plot(*P[0, :2], "o", c="tab:green", ms=9, label="start")
    if args.goal:
        ax.plot(gx, gy, "*", c="gold", mec="k", ms=16, label="goal")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Nav2 goal: {'PASS' if summary['passed'] else 'FAIL'}  ({elapsed:.0f} s, {path_len:.1f} m)")
    fig.savefig(args.out + ".png", dpi=100, bbox_inches="tight")

    print(json.dumps(summary, indent=2))
    print("PASS" if summary["passed"] else "FAIL")
    nav.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
