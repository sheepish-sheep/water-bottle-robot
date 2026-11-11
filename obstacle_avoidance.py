"""
Reactive obstacle avoidance for Stretch (no map needed).
Reads the lidar (LaserScan), drives toward a goal heading, and
slows / stops / turns away when something is in front.

Run INSIDE the humble container:
    distrobox enter humble
    cd ~/ament_ws && source install/setup.bash
    # terminal 1: launch the sim (stretch_simulation, like the workshop)
    # terminal 2: put the base in navigation mode, then run this
    python3 obstacle_avoidance.py
    # it stows the arm and aims the head first; to skip that:
    python3 obstacle_avoidance.py --ros-args -p stow_on_start:=false
    # override params if your topics are named differently:
    python3 obstacle_avoidance.py --ros-args -p scan_topic:=/scan_filtered -p cmd_topic:=/stretch/cmd_vel
    # also run depth_to_scan.py in another terminal so the head camera is used

Check topic names first with:  ros2 topic list | grep -E "scan|cmd_vel"

NOTE: on Stretch the lidar is mounted backwards (its 0 angle points at the
robot's rear). Nothing is assumed: at startup the node reads the lidar's
mounting angle from tf (base_link -> lidar frame) and uses that. You can still
force a value with -p angle_offset:=<radians>.
"""
import math

import numpy as np

# ---------- pure logic (no ROS, testable anywhere) ----------

DEFAULTS = dict(
    stop_dist=0.45,      # m, closer than this in front -> don't drive forward
    slow_dist=1.0,       # m, start slowing down inside this
    front_half_angle=math.radians(30),  # front sector is +-30 deg
    side_angle=math.radians(90),        # left/right sectors go out to +-90 deg
    max_v=0.4,           # m/s (sim needs more than 0.15 to actually move)
    max_w=1.2,           # rad/s
    k_heading=1.2,       # turn gain toward goal heading
    side_slow=0.45,      # m, something at the side closer than this -> steer away
    side_stop=0.30,      # m, closer than this at the side -> creep + hard turn away
    rear_stop=0.40,      # m, don't back up if something is this close behind
)


def rear_min(ranges, angles, half=math.radians(40)):
    """Closest thing behind the robot (within +-half of 180 deg)."""
    return min(sector_min(ranges, angles, math.pi - half, math.pi),
               sector_min(ranges, angles, -math.pi, -math.pi + half))


def stamp_sec(msg):
    """Sim time of a message (header stamp) in seconds, or None if unset.
    Using the sim's clock instead of the laptop's matters: with cameras on,
    the sim can run much slower than real time."""
    st = msg.header.stamp
    t = st.sec + st.nanosec * 1e-9
    return t if t > 0 else None


def wrap(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def sector_min(ranges, angles, lo, hi):
    """Min valid range with angle in [lo, hi]. inf if none."""
    mask = (angles >= lo) & (angles <= hi)
    vals = ranges[mask]
    return float(vals.min()) if vals.size else float("inf")


def clean_scan(ranges, angle_min, angle_inc, range_min, range_max, angle_offset=0.0):
    """Turn a raw scan into (ranges, angles) in the robot's frame, dropping bad readings."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + angle_inc * np.arange(r.size) + angle_offset
    a = (a + math.pi) % (2 * math.pi) - math.pi
    ok = np.isfinite(r) & (r > range_min) & (r < range_max)
    return r[ok], a[ok]


def decide(ranges, angles, goal_heading, p=None, turn_dir=0):
    """
    ranges, angles: cleaned scan, angle 0 = robot front, +ve = left
    goal_heading: direction we want to go, relative to robot front (rad)
    turn_dir: turn direction we're committed to (+1 left, -1 right, 0 none),
              pass back in what the last call returned
    Returns (v, w, state, front_dist, turn_dir)
    """
    p = dict(DEFAULTS, **(p or {}))
    fh, sa = p["front_half_angle"], p["side_angle"]
    front = sector_min(ranges, angles, -fh, fh)
    left = sector_min(ranges, angles, fh, sa)
    right = sector_min(ranges, angles, -sa, -fh)

    # already turning out of a blocked spot: keep turning the SAME way
    # until there's real room ahead (slow_dist, not just stop_dist).
    # Without this it flip-flops left/right forever in corners.
    if turn_dir != 0 and front < p["slow_dist"]:
        state = "TURNING_LEFT" if turn_dir > 0 else "TURNING_RIGHT"
        return 0.0, turn_dir * p["max_w"], state, front, turn_dir

    if front < p["stop_dist"]:
        # newly blocked: pick the side with more room and commit to it
        turn_dir = 1 if left >= right else -1
        state = "BLOCKED_TURN_LEFT" if turn_dir > 0 else "BLOCKED_TURN_RIGHT"
        return 0.0, turn_dir * p["max_w"], state, front, turn_dir

    turn_dir = 0
    # default: steer toward the goal
    w = max(-p["max_w"], min(p["max_w"], p["k_heading"] * wrap(goal_heading)))

    if front < p["slow_dist"]:
        # something ahead: slow down linearly and bias away from the closer side
        scale = (front - p["stop_dist"]) / (p["slow_dist"] - p["stop_dist"])
        v = p["max_v"] * scale
        w += 0.5 * p["max_w"] * (1 if left >= right else -1) * (1 - scale)
        w = max(-p["max_w"], min(p["max_w"], w))
        state = "SLOW"
    else:
        v = p["max_v"]
        state = "CLEAR"

    # side clearance: the front cone doesn't see a counter we're driving
    # alongside, and the arm/gripper hangs off the side of the base
    left_c = sector_min(ranges, angles, fh, math.radians(100))
    right_c = sector_min(ranges, angles, -math.radians(100), -fh)
    side_c = min(left_c, right_c)
    if side_c < p["side_slow"]:
        away = -1 if left_c < right_c else 1  # turn away from the closer side
        push = (p["side_slow"] - side_c) / (p["side_slow"] - p["side_stop"])
        push = max(0.0, min(1.0, push))
        w = max(-p["max_w"], min(p["max_w"], w + away * p["max_w"] * push))
        v *= max(0.3, 1 - push)
        if side_c < p["side_stop"]:
            v = min(v, 0.03)
            w = away * p["max_w"]
        state += "+SIDE_L" if left_c < right_c else "+SIDE_R"

    # don't drive forward hard while the goal is way off to the side
    if abs(wrap(goal_heading)) > math.radians(60):
        v = min(v, 0.3 * p["max_v"])
    return v, w, state, front, turn_dir


# ---------- ROS 2 node ----------

def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from rclpy.time import Time
    import time

    class Avoider(Node):
        def __init__(self):
            super().__init__("obstacle_avoidance")
            self.declare_parameter("scan_topic", "/scan_filtered")
            self.declare_parameter("cmd_topic", "/stretch/cmd_vel")
            # lidar mounting yaw relative to base_link. NaN = read it from tf (default)
            self.declare_parameter("angle_offset", float("nan"))
            self.declare_parameter("base_frame", "base_link")
            # smallest command that actually moves the robot (the sim robot just
            # slips at ~0.15 m/s); nonzero commands below this get bumped up.
            # 0 disables.
            self.declare_parameter("min_v", 0.25)  # m/s
            self.declare_parameter("min_w", 0.6)   # rad/s, only applied when barely driving
            self.declare_parameter("goal_heading", 0.0)  # rad, 0 = straight ahead
            self.declare_parameter("odom_topic", "/odom")
            # in an enclosed room the lidar always hits a wall; NO returns at all
            # straight ahead/behind means open edge of the world -> treat as an
            # obstacle this far away. 0 disables.
            self.declare_parameter("void_dist", 0.0)  # OFF by default; sim-only shortcut, e.g. 0.40
            # startup: stow the arm and point the head forward/down before driving.
            # The arm sticking out sideways hooks on counters, and the head camera
            # only helps if it looks where the robot is going.
            self.declare_parameter("stow_on_start", True)
            self.declare_parameter("stow_service", "/stow_the_robot")
            self.declare_parameter("head_action", "/stretch_controller/follow_joint_trajectory")
            self.declare_parameter("head_pan", 0.0)
            self.declare_parameter("head_tilt", -0.7)   # rad; more negative = looks closer to the base
            self.declare_parameter("settle_time", 2.0)  # s to wait after setup before driving
            # stowing / moving the head can switch the driver to position mode,
            # where it ignores cmd_vel -> switch back before driving
            self.declare_parameter("nav_mode_service", "/switch_to_navigation_mode")
            # virtual scan from the head depth camera (depth_to_scan.py); "" to disable
            self.declare_parameter("depth_scan_topic", "/scan_depth")
            # stuck detection: told to drive but odom barely moved
            self.declare_parameter("stuck_time", 2.0)     # s of trying before calling it stuck
            self.declare_parameter("stuck_dist", 0.02)    # m, ahead/behind distance changed less than this = stuck
            self.declare_parameter("backup_time", 1.5)    # s reversing
            self.declare_parameter("backup_speed", 0.08)  # m/s
            self.declare_parameter("escape_turn_time", 2.5)  # s turning after backing up
            for k, v in DEFAULTS.items():
                self.declare_parameter(k, float(v))

            g = lambda n: self.get_parameter(n).value
            self.cmd_pub = self.create_publisher(Twist, g("cmd_topic"), 10)
            self.create_subscription(LaserScan, g("scan_topic"), self.on_scan, qos_profile_sensor_data)
            self.last_scan_time = self.get_clock().now()
            self.create_timer(0.2, self.watchdog)
            self.create_subscription(Odometry, g("odom_topic"), self.on_odom, 10)
            self.depth = None  # (time, ranges, angles) from the depth camera
            if g("depth_scan_topic"):
                self.create_subscription(LaserScan, g("depth_scan_topic"), self.on_depth_scan, 10)
            self.n = 0
            self.turn_dir = 0
            self.ready = False  # no driving until setup() is done
            self.offset = None  # lidar yaw in base_link, resolved on first scan
            self.tf_fail = 0
            from tf2_ros import Buffer, TransformListener
            self.tf_buf = Buffer()
            self.tf_listener = TransformListener(self.tf_buf, self)
            self.pos = None            # latest (x, y) from odom (just for info)
            self.last_change = 0.0
            self.drive_start = None    # (time, scan) when we started commanding forward
            self.recovery = None       # None, or (phase, end_time, turn_sign)
            self.last_lr = (0.0, 0.0)  # (left, right) clearance, for choosing escape side
            self.get_logger().info(f"listening on {g('scan_topic')}, publishing {g('cmd_topic')}")

        def params(self):
            return {k: self.get_parameter(k).value for k in DEFAULTS}

        def setup(self):
            """Stow the arm and aim the head. Blocks until done (or timed out)."""
            from std_srvs.srv import Trigger
            from control_msgs.action import FollowJointTrajectory
            from trajectory_msgs.msg import JointTrajectoryPoint
            from builtin_interfaces.msg import Duration
            from rclpy.action import ActionClient

            g = lambda n: self.get_parameter(n).value
            if g("stow_on_start"):
                cli = self.create_client(Trigger, g("stow_service"))
                if cli.wait_for_service(timeout_sec=3.0):
                    self.get_logger().info("stowing the robot...")
                    fut = cli.call_async(Trigger.Request())
                    rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
                    res = fut.result()
                    if res is None:
                        self.get_logger().warn("stow call timed out, continuing anyway")
                    elif not res.success:
                        self.get_logger().warn(f"stow failed: {res.message}")
                    else:
                        self.get_logger().info("stowed")
                else:
                    self.get_logger().warn(f"{g('stow_service')} not available, skipping stow")

            ac = ActionClient(self, FollowJointTrajectory, g("head_action"))
            if ac.wait_for_server(timeout_sec=3.0):
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = ["joint_head_pan", "joint_head_tilt"]
                pt = JointTrajectoryPoint()
                pt.positions = [float(g("head_pan")), float(g("head_tilt"))]
                pt.time_from_start = Duration(sec=2)
                goal.trajectory.points = [pt]
                self.get_logger().info(f"aiming head: pan={g('head_pan')} tilt={g('head_tilt')}")
                fut = ac.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
                gh = fut.result()
                if gh is not None and gh.accepted:
                    rf = gh.get_result_async()
                    rclpy.spin_until_future_complete(self, rf, timeout_sec=10.0)
                    self.get_logger().info("head aimed")
                else:
                    self.get_logger().warn("head goal rejected or timed out, continuing anyway")
            else:
                self.get_logger().warn(f"{g('head_action')} not available, skipping head aim")

            cli = self.create_client(Trigger, g("nav_mode_service"))
            if cli.wait_for_service(timeout_sec=3.0):
                fut = cli.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
                res = fut.result()
                if res is not None and res.success:
                    self.get_logger().info("switched to navigation mode")
                else:
                    self.get_logger().warn(f"switch to navigation mode failed: {res.message if res else 'timeout'}")
            else:
                self.get_logger().warn(f"{g('nav_mode_service')} not available - if the robot won't drive, "
                                       "check `ros2 topic echo /mode --once`")

            time.sleep(float(g("settle_time")))
            self.ready = True
            self.get_logger().info("setup done, driving")

        def resolve_offset(self, frame):
            """Lidar yaw in base_link: from the parameter if set, else from tf."""
            if self.offset is not None:
                return True
            p = self.get_parameter("angle_offset").value
            if p is not None and not math.isnan(p):
                self.offset = float(p)
                self.get_logger().info(f"lidar angle_offset from parameter: {math.degrees(self.offset):.1f} deg")
                return True
            try:
                t = self.tf_buf.lookup_transform(self.get_parameter("base_frame").value, frame, Time())
                q = t.transform.rotation
                self.offset = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                self.get_logger().info(f"lidar angle_offset from tf ({frame} in base_link): "
                                       f"{math.degrees(self.offset):.1f} deg")
                return True
            except Exception as e:
                self.tf_fail += 1
                if self.tf_fail % 20 == 1:
                    self.get_logger().warn(f"waiting for tf {frame} -> base_link to get lidar mounting angle: {e}")
                return False

        def on_depth_scan(self, msg):
            r, a = clean_scan(msg.ranges, msg.angle_min, msg.angle_increment,
                              msg.range_min, msg.range_max,
                              self.offset if self.offset is not None else 0.0)
            # freshness = "is depth_to_scan alive", so use arrival time on this
            # machine, not the stamp (camera and lidar stamps can use different clocks)
            self.depth = (time.monotonic(), r, a)

        def on_odom(self, msg):
            p = msg.pose.pose.position
            self.pos = (p.x, p.y)

        def check_stuck(self, v, w, now, ahead, behind):
            """
            True if we've been commanding motion (driving OR turning) for stuck_time but the distance
            straight ahead / straight behind didn't change. Uses lidar, not odom,
            because in the sim the wheels can spin against a counter and odom
            still says we're moving. Only the beams straight ahead/behind are used:
            in a hallway the side-wall beams don't change even when driving.
            """
            g = lambda n: self.get_parameter(n).value
            if abs(v) < 0.03 and abs(w) < 0.3:
                self.drive_start = None
                return False
            if self.drive_start is None:
                self.drive_start = (now, ahead, behind)
                return False
            t0, a0, b0 = self.drive_start
            if now - t0 < g("stuck_time"):
                return False
            changes = [abs(x1 - x0) for x0, x1 in ((a0, ahead), (b0, behind))
                       if math.isfinite(x0) and math.isfinite(x1)]
            self.drive_start = (now, ahead, behind)  # start a new window
            if not changes:
                return False  # nothing in range ahead or behind, can't tell
            self.last_change = max(changes)
            return self.last_change < g("stuck_dist")

        def on_scan(self, msg):
            self.last_scan_time = self.get_clock().now()
            if not self.ready or not self.resolve_offset(msg.header.frame_id):
                self.cmd_pub.publish(Twist())
                return
            now = stamp_sec(msg) or time.monotonic()
            g = lambda n: self.get_parameter(n).value
            r, a = clean_scan(msg.ranges, msg.angle_min, msg.angle_increment,
                              msg.range_min, msg.range_max, self.offset)
            # merge in depth-camera points if they're fresh (< 0.5 s old)
            # "void" check on the lidar alone, before merging camera points:
            # any beam ahead / straight behind with NO return -> open edge that way
            void = ""
            vd = g("void_dist")
            if vd > 0:
                raw = np.asarray(msg.ranges, dtype=float)
                ra = msg.angle_min + msg.angle_increment * np.arange(raw.size) + self.offset
                ra = (ra + math.pi) % (2 * math.pi) - math.pi
                no_ret = ~np.isfinite(raw) | (raw <= msg.range_min) | (raw >= msg.range_max)
                fh = self.get_parameter("front_half_angle").value
                # rear cone kept narrow (+-25 deg) so it doesn't overlap the mast's blanked wedge
                ahead_void = no_ret & (np.abs(ra) <= fh)
                behind_void = no_ret & (np.abs(ra) >= math.radians(155))
                if ahead_void.any():
                    void += "+VOID_AHEAD"
                if behind_void.any():
                    void += "+VOID_BEHIND"
                fa = ra[ahead_void | behind_void]
                if fa.size:
                    r = np.concatenate([r, np.full(fa.size, vd)])
                    a = np.concatenate([a, fa])

            n_depth = 0
            if self.depth is not None and time.monotonic() - self.depth[0] < 1.0:
                r = np.concatenate([r, self.depth[1]])
                a = np.concatenate([a, self.depth[2]])
                n_depth = self.depth[1].size
            p = self.params()
            left = sector_min(r, a, p["front_half_angle"], math.radians(100))
            right = sector_min(r, a, -math.radians(100), -p["front_half_angle"])

            front = sector_min(r, a, -p["front_half_angle"], p["front_half_angle"])
            rear = rear_min(r, a)

            if self.recovery is not None:
                phase, end, sign = self.recovery
                # stop backing up early if we're about to hit something behind
                if phase == "BACKUP" and (now >= end or rear < p["rear_stop"]):
                    self.recovery = ("ESCAPE_TURN", now + g("escape_turn_time"), sign)
                elif phase == "ESCAPE_TURN" and now >= end:
                    self.recovery = None
                    self.turn_dir = 0
                    self.drive_start = None
            if self.recovery is not None:
                phase, end, sign = self.recovery
                if phase == "BACKUP":
                    v, w, state = -g("backup_speed"), 0.0, "STUCK_BACKING_UP"
                else:
                    # spinning in place next to a counter swings the arm into it,
                    # so creep forward while turning if there's room ahead
                    v = 0.05 if front > p["stop_dist"] else 0.0
                    w = sign * p["max_w"]
                    state = "STUCK_TURN_LEFT" if sign > 0 else "STUCK_TURN_RIGHT"
            else:
                v, w, state, front, self.turn_dir = decide(
                    r, a, g("goal_heading"), p, self.turn_dir)
                narrow = math.radians(5)
                ahead = sector_min(r, a, -narrow, narrow)
                behind = min(sector_min(r, a, math.pi - narrow, math.pi),
                             sector_min(r, a, -math.pi, -math.pi + narrow))
                if self.check_stuck(v, w, now, ahead, behind):
                    sign = 1 if left >= right else -1
                    if rear < p["rear_stop"]:
                        # no room behind: skip backing up, go straight to the escape turn
                        self.recovery = ("ESCAPE_TURN", now + g("escape_turn_time"), sign)
                        self.get_logger().warn(f"STUCK ({self.last_change:.3f} m change), wall behind -> turning out")
                        v, w, state = 0.0, sign * p["max_w"], "STUCK_TURN_LEFT" if sign > 0 else "STUCK_TURN_RIGHT"
                    else:
                        self.recovery = ("BACKUP", now + g("backup_time"), sign)
                        self.get_logger().warn(f"STUCK ({self.last_change:.3f} m change) -> backing up")
                        v, w, state = -g("backup_speed"), 0.0, "STUCK_BACKING_UP"
            # deadband: tiny nonzero commands don't move the robot, bump them up
            mv, mw = g("min_v"), g("min_w")
            if mv > 0 and 1e-3 < abs(v) < mv:
                v = math.copysign(mv, v)
            if mw > 0 and abs(v) < mv and 1e-3 < abs(w) < mw:
                w = math.copysign(mw, w)
            t = Twist()
            t.linear.x, t.angular.z = float(v), float(w)
            self.cmd_pub.publish(t)
            self.n += 1
            if self.n % 10 == 0:
                state = state + void
                self.get_logger().info(f"{state:22s} front={front:.2f}m left={left:.2f}m right={right:.2f}m rear={rear:.2f}m depth_pts={n_depth} "
                                       f"v={v:.2f} w={w:.2f}")

        def watchdog(self):
            # no lidar for 0.5 s -> stop, don't drive blind
            if (self.get_clock().now() - self.last_scan_time).nanoseconds > 5e8:
                self.cmd_pub.publish(Twist())

        def stop(self):
            self.cmd_pub.publish(Twist())

    rclpy.init()
    node = Avoider()
    try:
        node.setup()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
