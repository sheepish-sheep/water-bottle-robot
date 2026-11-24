#!/usr/bin/env python3
"""
Drive Stretch to the NEAREST water fountain from wherever it starts.

Nav2 plans the route on the map; by default our own reactive obstacle
avoidance (water_bottle_robot/reactive/obstacle_avoidance.py's decide()) does the driving.

Needs Nav2 running first (on the robot: the class repo's
`ros2 launch parcs_stretch_nav2 navigation.launch.py teleop_type:=keyboard`).

1) Record fountains once. Saved to ~/.ros/water_bottle_robot/fountains.yaml
   (or $WBR_FOUNTAINS_FILE). Either:
   a) On the robot (most accurate): localize, drive it with teleop to where it
      should stop, FACING the fountain, then
          ros2 run water_bottle_robot go_to_fountain --add-here east_hall
      This saves the robot's current pose from AMCL/tf.
   b) From the map only (RViz "Publish Point" tool, two clicks):
          ros2 run water_bottle_robot go_to_fountain --add east_hall
      click 1 = floor spot where the robot should STOP (~0.6 m out from the fountain)
      click 2 = the fountain itself (sets which way the robot faces at the end)
   Repeat for each fountain. Re-using a name overwrites that fountain.

2) Go:
       ros2 run water_bottle_robot go_to_fountain                       # global localization + reactive driving
       ros2 run water_bottle_robot go_to_fountain --localize manual     # you set "2D Pose Estimate" in RViz first
       ros2 run water_bottle_robot go_to_fountain --controller nav2     # let Nav2's own controller drive instead

How it works:
  * Localize: --localize global calls AMCL's /reinitialize_global_localization
    (guesses spread over the whole map) and spins slowly in place until the
    pose uncertainty is small. Long similar hallways can fool it; if so, set
    the pose by hand in RViz and use --localize manual.
  * Nearest fountain = shortest PLANNED path (Nav2 planner), not straight-line
    distance, so walls are accounted for.
  * --controller reactive: every lidar scan, take a point on the path
    --lookahead m ahead of the robot, turn it into a heading relative to the
    robot's front, and hand that to decide() as goal_heading. decide() does the
    obstacle avoidance. Replans if the robot ends up --replan-dist m off the path
    or gets stuck. Nav2's own controller is never sent a goal in this mode, so
    the two don't fight over cmd_vel.
  * --controller nav2: sends the fountain to Nav2 (goToPose); Nav2 avoids obstacles.
  * At the end, turns in place to face the fountain.

Ctrl+C stops the robot.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

from water_bottle_robot.reactive.obstacle_avoidance import DEFAULTS, clean_scan, decide, rear_min, wrap

# Outside the package: an installed package's own folder gets overwritten on every rebuild.
FOUNTAINS_FILE = os.environ.get("WBR_FOUNTAINS_FILE",
                                os.path.expanduser("~/.ros/water_bottle_robot/fountains.yaml"))


# ---------- pure geometry (no ROS, testable anywhere) ----------

def path_length(pts):
    """pts: Nx2 array. Total length in m."""
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def closest_index(pts, x, y, start=0):
    """Index of the path point closest to (x, y), searching from `start` on
    (so the robot never jumps back to an earlier part of the path)."""
    d = np.hypot(pts[start:, 0] - x, pts[start:, 1] - y)
    i = int(np.argmin(d))
    return start + i, float(d[i])


def lookahead_point(pts, i0, x, y, dist):
    """First path point at/after index i0 that is at least `dist` from the
    robot; the path's last point if none is that far."""
    d = np.hypot(pts[i0:, 0] - x, pts[i0:, 1] - y)
    far = np.nonzero(d >= dist)[0]
    j = i0 + int(far[0]) if far.size else len(pts) - 1
    return pts[j]


def heading_to(px, py, x, y, yaw):
    """Direction of (px, py) relative to the robot's front, radians (+ = left)."""
    return wrap(math.atan2(py - y, px - x) - yaw)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_fountains(path=FOUNTAINS_FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("fountains", []) or []


def save_fountains(fountains, path=FOUNTAINS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"fountains": fountains}, f, sort_keys=False)


# ---------- ROS ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", metavar="NAME", help="record a fountain from two RViz Publish Point clicks")
    ap.add_argument("--add-here", metavar="NAME",
                    help="record a fountain at the robot's current pose (park it facing the fountain first)")
    ap.add_argument("--controller", choices=["reactive", "nav2"], default="reactive")
    ap.add_argument("--localize", choices=["global", "manual", "none"], default="global",
                    help="none = pose is already known (e.g. the sim, which has no AMCL)")
    ap.add_argument("--max-v", type=float, default=0.2, help="m/s (reactive mode)")
    ap.add_argument("--min-v", type=float, default=0.05, help="m/s, smallest forward command that moves the robot")
    ap.add_argument("--lookahead", type=float, default=0.8, help="m")
    ap.add_argument("--arrive-tol", type=float, default=0.3, help="m from the stop spot counts as arrived")
    ap.add_argument("--replan-dist", type=float, default=1.0, help="m off the path before replanning")
    ap.add_argument("--timeout", type=float, default=300.0, help="s for the whole drive")
    ap.add_argument("--scan-topic", default="/scan_filtered")
    ap.add_argument("--cmd-topic", default="/stretch/cmd_vel")
    ap.add_argument("--no-stow", action="store_true")
    args = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
    from lifecycle_msgs.srv import GetState
    from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty, Trigger
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    nav = BasicNavigator()
    log = nav.get_logger()

    # ----- record mode -----
    if args.add:
        clicks = []
        nav.create_subscription(PointStamped, "/clicked_point", lambda m: clicks.append(m), 10)
        print(f"In RViz, pick 'Publish Point' and click where the robot should STOP for '{args.add}' "
              "(about 0.6 m out from the fountain)...")
        while len(clicks) < 1:
            rclpy.spin_once(nav, timeout_sec=0.2)
        print(f"  stop spot: ({clicks[0].point.x:.2f}, {clicks[0].point.y:.2f}). "
              "Now 'Publish Point' again and click the fountain itself...")
        while len(clicks) < 2:
            rclpy.spin_once(nav, timeout_sec=0.2)
        s, f = clicks[0].point, clicks[1].point
        if clicks[0].header.frame_id != "map":
            print(f"WARNING: clicks were in frame '{clicks[0].header.frame_id}', expected 'map'. "
                  "Set RViz's Fixed Frame to map.")
        entry = {"name": args.add, "x": round(s.x, 3), "y": round(s.y, 3),
                 "yaw": round(math.atan2(f.y - s.y, f.x - s.x), 3)}
        fountains = [e for e in load_fountains() if e["name"] != args.add] + [entry]
        save_fountains(fountains)
        print(f"saved {entry} to {FOUNTAINS_FILE} ({len(fountains)} fountain(s) total)")
        rclpy.shutdown()
        return

    # ----- record at the robot's current pose -----
    if args.add_here:
        buf = Buffer()
        TransformListener(buf, nav)
        print("reading the robot's pose (map -> base_link)...")
        pose, end = None, time.monotonic() + 15.0
        while pose is None and time.monotonic() < end:
            rclpy.spin_once(nav, timeout_sec=0.2)
            try:
                t = buf.lookup_transform("map", "base_link", Time())
                pose = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
            except Exception:
                pass
        if pose is None:
            rclpy.shutdown()
            sys.exit("No map -> base_link transform. Is Nav2/AMCL running and the robot localized?")
        entry = {"name": args.add_here, "x": round(pose[0], 3), "y": round(pose[1], 3), "yaw": round(pose[2], 3)}
        fountains = [e for e in load_fountains() if e["name"] != args.add_here] + [entry]
        save_fountains(fountains)
        print(f"saved {entry} to {FOUNTAINS_FILE} ({len(fountains)} fountain(s) total)")
        print("Check in RViz that the robot model sits where it really is before trusting this.")
        rclpy.shutdown()
        return

    fountains = load_fountains()
    if not fountains:
        sys.exit(f"No fountains in {FOUNTAINS_FILE}. Record some first with --add NAME.")

    cmd_pub = nav.create_publisher(Twist, args.cmd_topic, 10)
    tf_buf = Buffer()
    TransformListener(tf_buf, nav)
    scan = {}       # latest cleaned scan
    lidar = {}      # lidar yaw in base_link (from tf)
    amcl = {}       # latest /amcl_pose

    def on_scan(m):
        if "off" not in lidar:
            try:
                t = tf_buf.lookup_transform("base_link", m.header.frame_id, Time())
                lidar["off"] = yaw_of(t.transform.rotation)
                log.info(f"lidar mounted at {math.degrees(lidar['off']):.0f} deg in base_link")
            except Exception:
                return
        r, a = clean_scan(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, lidar["off"])
        scan.update(r=r, a=a, t=time.monotonic())

    nav.create_subscription(LaserScan, args.scan_topic, on_scan, qos_profile_sensor_data)
    nav.create_subscription(
        PoseWithCovarianceStamped, "/amcl_pose", lambda m: amcl.update(msg=m, t=time.monotonic()),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))

    def spin_for(sec):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(nav, timeout_sec=0.05)

    def stop():
        cmd_pub.publish(Twist())

    def drive(v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        cmd_pub.publish(t)

    def robot_pose():
        """(x, y, yaw) of base_link in the map, or None."""
        try:
            t = tf_buf.lookup_transform("map", "base_link", Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def call_trigger(name, timeout=20.0):
        cli = nav.create_client(Trigger, name)
        if not cli.wait_for_service(timeout_sec=3.0):
            log.warn(f"{name} not available, skipping")
            return
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(nav, fut, timeout_sec=timeout)
        res = fut.result()
        log.info(f"{name}: {res.message if res else 'timed out'}")

    def wait_active(node, timeout=120.0):
        cli = nav.create_client(GetState, f"/{node}/get_state")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cli.wait_for_service(timeout_sec=1.0):
                fut = cli.call_async(GetState.Request())
                rclpy.spin_until_future_complete(nav, fut, timeout_sec=2.0)
                if fut.result() is not None and fut.result().current_state.label == "active":
                    return True
            log.info(f"waiting for {node} to be active (pressed Startup in RViz?)", throttle_duration_sec=5.0)
            spin_for(1.0)
        return False

    def pose_stamped(x, y, yaw):
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = nav.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return p

    def plan_to(f):
        """Nav2 planner path from the robot's current pose to fountain f, as Nx2 array (or None)."""
        cur = robot_pose()
        if cur is None:
            return None
        path = nav.getPath(pose_stamped(*cur), pose_stamped(f["x"], f["y"], f["yaw"]), use_start=True)
        if path is None or not path.poses:
            return None
        return np.array([[p.pose.position.x, p.pose.position.y] for p in path.poses])

    def clearance():
        return float(scan["r"].min()) if scan.get("r") is not None and scan["r"].size else float("nan")

    # ----- 1. setup -----
    try:
        for node in (["planner_server"] + (["amcl"] if args.localize != "none" else [])
                     + (["bt_navigator"] if args.controller == "nav2" else [])):
            if not wait_active(node):
                sys.exit(f"{node} never became active. Is Nav2 running?")
        if not args.no_stow:
            call_trigger("/stow_the_robot")
        call_trigger("/switch_to_navigation_mode")
        spin_for(1.0)

        # ----- 2. localize -----
        if args.localize == "global":
            cli = nav.create_client(Empty, "/reinitialize_global_localization")
            if not cli.wait_for_service(timeout_sec=5.0):
                sys.exit("/reinitialize_global_localization not available (is AMCL running?)")
            if not scan:
                spin_for(2.0)
            if clearance() < 0.35:
                sys.exit(f"Something is {clearance():.2f} m away. Move the robot into open space "
                         "so it can spin in place, then rerun.")
            rclpy.spin_until_future_complete(nav, cli.call_async(Empty.Request()), timeout_sec=5.0)
            log.info("global localization: spinning in place until AMCL is confident...")
            amcl.clear()
            start, ok = time.monotonic(), False
            while time.monotonic() - start < 60.0:
                drive(0.0, 0.5)
                rclpy.spin_once(nav, timeout_sec=0.1)
                m = amcl.get("msg")
                if m is not None and time.monotonic() - start > 6.0:  # at least ~half a turn of data
                    c = m.pose.covariance
                    sx, sy, syaw = math.sqrt(max(c[0], 0)), math.sqrt(max(c[7], 0)), math.sqrt(max(c[35], 0))
                    log.info(f"  pose std: x={sx:.2f} m y={sy:.2f} m yaw={math.degrees(syaw):.0f} deg",
                             throttle_duration_sec=2.0)
                    if sx < 0.25 and sy < 0.25 and syaw < 0.25:
                        ok = True
                        break
            stop()
            if not ok:
                sys.exit("AMCL didn't converge. Set '2D Pose Estimate' in RViz and rerun with --localize manual.")
            p = amcl["msg"].pose.pose
            log.info(f"localized at ({p.position.x:.2f}, {p.position.y:.2f})")
        elif args.localize == "manual":
            log.info("waiting for a pose from AMCL (set '2D Pose Estimate' in RViz)...")
            while "msg" not in amcl:
                rclpy.spin_once(nav, timeout_sec=0.2)
        while robot_pose() is None:
            log.info("waiting for map -> base_link tf...", throttle_duration_sec=3.0)
            rclpy.spin_once(nav, timeout_sec=0.2)

        # ----- 3. nearest fountain by planned path length -----
        best = None
        for f in fountains:
            pts = plan_to(f)
            if pts is None:
                log.warn(f"  {f['name']}: no path")
                continue
            length = path_length(pts)
            log.info(f"  {f['name']}: {length:.1f} m")
            if best is None or length < best[1]:
                best = (f, length, pts)
        if best is None:
            sys.exit("No fountain is reachable from here. Check the robot's pose in RViz.")
        goal, length, pts = best
        log.info(f"going to '{goal['name']}' ({length:.1f} m away)")

        # ----- 4. drive -----
        if args.controller == "nav2":
            nav.goToPose(pose_stamped(goal["x"], goal["y"], goal["yaw"]))
            start = time.monotonic()
            while not nav.isTaskComplete():
                fb = nav.getFeedback()
                if fb:
                    log.info(f"distance left: {fb.distance_remaining:.2f} m", throttle_duration_sec=2.0)
                if time.monotonic() - start > args.timeout:
                    nav.cancelTask()
                    sys.exit("timed out")
            if nav.getResult() != TaskResult.SUCCEEDED:
                sys.exit(f"Nav2 didn't reach the fountain: {nav.getResult()}")
        else:
            p = dict(DEFAULTS, max_v=args.max_v)
            i0 = 0
            turn_dir = 0
            moved = None          # (time, x, y) for stuck detection
            backup_until = 0.0
            start = time.monotonic()
            while True:
                rclpy.spin_once(nav, timeout_sec=0.05)
                now = time.monotonic()
                if now - start > args.timeout:
                    stop()
                    sys.exit("timed out")
                cur = robot_pose()
                if cur is None or not scan or now - scan["t"] > 0.5:
                    stop()  # no pose or no fresh lidar: don't drive blind
                    continue
                x, y, yaw = cur
                if math.hypot(goal["x"] - x, goal["y"] - y) < args.arrive_tol:
                    stop()
                    break

                if now < backup_until:  # stuck recovery: back up (unless something is behind)
                    if rear_min(scan["r"], scan["a"]) < p["rear_stop"]:
                        backup_until = 0.0
                    else:
                        drive(-0.08, 0.0)
                        continue

                i0, off_path = closest_index(pts, x, y, i0)
                if off_path > args.replan_dist:
                    log.info(f"{off_path:.1f} m off the path, replanning")
                    new = plan_to(goal)
                    if new is not None:
                        pts, i0 = new, 0
                        continue
                lx, ly = lookahead_point(pts, i0, x, y, args.lookahead)
                v, w, state, front, turn_dir = decide(scan["r"], scan["a"], heading_to(lx, ly, x, y, yaw),
                                                      p, turn_dir)
                if 1e-3 < abs(v) < args.min_v:
                    v = math.copysign(args.min_v, v)
                drive(v, w)
                log.info(f"{state:20s} front={front:.2f} m  to goal={math.hypot(goal['x'] - x, goal['y'] - y):.1f} m",
                         throttle_duration_sec=1.0)

                # stuck: asked to drive forward for 3 s but moved < 5 cm
                if v > 0.03:
                    if moved is None:
                        moved = (now, x, y)
                    elif now - moved[0] > 3.0:
                        if math.hypot(x - moved[1], y - moved[2]) < 0.05:
                            log.warn("stuck, backing up and replanning")
                            backup_until = now + 1.5
                            turn_dir = 0
                            new = plan_to(goal)
                            if new is not None:
                                pts, i0 = new, 0
                        moved = None
                else:
                    moved = None

        # ----- 5. face the fountain -----
        end = time.monotonic() + 15.0
        while time.monotonic() < end:
            rclpy.spin_once(nav, timeout_sec=0.05)
            cur = robot_pose()
            if cur is None:
                continue
            err = wrap(goal["yaw"] - cur[2])
            if abs(err) < math.radians(6):
                break
            drive(0.0, max(-0.6, min(0.6, 1.2 * err)) if abs(err) > 0.3 else math.copysign(0.3, err))
        stop()
        log.info(f"arrived at '{goal['name']}'")
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        spin_for(0.2)
        nav.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
