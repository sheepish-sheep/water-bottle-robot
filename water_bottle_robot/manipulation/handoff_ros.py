"""
Bottle handoff through ROS 2: the same fixed-spot handoff as handoff_real, but
it commands the robot through Hello Robot's driver (stretch_core's
stretch_driver on the robot, stretch_simulation's driver in MuJoCo) using
hello_helpers' HelloNode, instead of talking to stretch_body directly.

    # robot, terminal 1: the driver, in position mode
    ros2 launch stretch_core stretch_driver.launch.py mode:=position
    # robot, terminal 2
    ros2 run water_bottle_robot handoff_ros --distance 0.70 --hand-top 0.83 --dry-run
    ros2 run water_bottle_robot handoff_ros --distance 0.70 --hand-top 0.83 --tug-threshold 8 --tug-sign 1

Same options, safety stops and log as handoff_real (see docs/handoff_real_robot.md).
In sim (scripts/run_handoff_ros_sim.sh) only the dry run is meaningful: the sim
scene has no person or bottle, and the sim driver reports zero arm effort.

How commands go out: moves the state machine issues together (e.g. arm + lift)
are sent as ONE FollowJointTrajectory goal, because the driver runs one goal at
a time and rejects goals sent less than 0.2 s apart. Every joint gets an explicit
speed: a missing speed is sent as 0.0, which stretch_body passes to the wrist and
gripper motors, where it means "no speed limit".
"""
import argparse
import math
import sys
import threading
import time

from water_bottle_robot.manipulation.handoff_controller import RobotInterface
from water_bottle_robot.manipulation.handoff_real import (
    LOG_COLUMNS, Abort, add_handoff_args, confirm, hand_over_if_holding, new_log, plan_from_args, run)


class RosRobot(RobotInterface):
    """RobotInterface over stretch_driver's FollowJointTrajectory action and /stretch/joint_states."""

    dt = 0.05
    JOINTS = {"lift": "joint_lift", "arm": "wrist_extension", "wrist_yaw": "joint_wrist_yaw",
              "wrist_pitch": "joint_wrist_pitch", "wrist_roll": "joint_wrist_roll"}
    GRIPPER = "joint_gripper_finger_left"
    TOL = {"lift": 0.01, "arm": 0.01, "wrist_yaw": 0.05, "wrist_pitch": 0.05, "wrist_roll": 0.05}
    STILL = {"lift": 0.005, "arm": 0.005}   # m/s; wrist joints use 0.05 rad/s
    GOAL_RETRY_S = 3.0

    def __init__(self, node, spot, close_pct=-50.0, holding_pct=5.0, tug_sign=1.0, log=None,
                 gripper_speed=None):
        from hello_helpers.gripper_conversion import GripperConversion
        conv = GripperConversion()
        # same pct units as the stretch_body version, converted to the driver's finger angle (rad)
        self.open_rad = conv.robotis_to_finger(50.0)
        self.close_rad = conv.robotis_to_finger(close_pct)
        self.holding_rad = conv.robotis_to_finger(holding_pct)
        self.node, self.spot, self.tug_sign, self.log = node, spot, tug_sign, log
        self.gripper_speed = gripper_speed
        self.pending = {}          # joint name -> (position, speed); sent as one goal
        self.targets = {}
        self.grip_cmd = None
        self.phase = ""
        self.runstopped = False
        self._lock = threading.Lock()

    # --- state (from /stretch/joint_states) ---
    def _js(self, name):
        js = self.node.joint_state
        try:
            i = list(js.name).index(name)
        except ValueError:
            raise Abort(f"joint '{name}' isn't in /stretch/joint_states (driver running? joints: {list(js.name)})")
        pos = js.position[i]
        vel = js.velocity[i] if len(js.velocity) > i else 0.0
        eff = js.effort[i] if len(js.effort) > i else 0.0
        return pos, vel, eff

    def has_joint(self, name):
        return name in self.node.joint_state.name

    def joint(self, name):
        return float(self._js(self.JOINTS[name])[0])

    def arm_effort(self):
        # stretch_driver publishes the arm motor's effort_pct as the effort of wrist_extension
        return self.tug_sign * float(self._js("wrist_extension")[2])

    def finger(self):
        return float(self._js(self.GRIPPER)[0])

    def gripper_holding(self):
        return self.grip_cmd == "closed" and self.finger() > self.holding_rad

    def estimate_bottle(self):
        return dict(self.spot)

    def check_safety(self):
        if self.runstopped:
            raise Abort("run-stop pressed")

    # --- commands: collected, then sent as one goal when the state machine waits ---
    def move(self, joint, target, speed):
        name = self.JOINTS[joint]
        if not self.has_joint(name):
            return   # e.g. no wrist pitch/roll without the dex wrist
        self.pending[name] = (float(target), float(speed))
        self.targets[joint] = target

    def turn_base(self, angle, speed):
        if abs(angle) > math.radians(1.0):
            raise Abort(f"plan wants a {math.degrees(angle):.1f} deg base turn; base motion is disabled here")

    def set_gripper(self, opening):
        self.grip_cmd = opening
        self.pending[self.GRIPPER] = (self.open_rad if opening == "open" else self.close_rad, self.gripper_speed)

    def flush(self):
        if not self.pending:
            return
        pose = dict(self.pending)
        self.pending.clear()
        speeds = [v for _, v in pose.values()]
        if any(v is None for v in speeds):
            if len(pose) > 1:
                raise Abort(f"can't send joints with and without speeds in one goal: {pose}")
            goal = {k: p for k, (p, _) in pose.items()}          # gripper alone: driver default speed
        else:
            goal = {k: (p, v) for k, (p, v) in pose.items()}     # every joint has an explicit speed
        t0 = time.time()
        while True:
            fut = self.node.move_to_pose(goal, blocking=False)
            while fut is not None and not fut.done():
                time.sleep(0.01)
            handle = fut.result() if fut is not None else None
            if handle is not None and handle.accepted:
                return
            if time.time() - t0 > self.GOAL_RETRY_S:
                raise Abort(f"driver rejected the goal {goal} (is it in position mode?)")
            time.sleep(0.25)   # the driver rejects goals < 0.2 s after the previous one

    def halt(self):
        self.pending.clear()
        try:
            self.node.stop_the_robot()
        except Exception as e:
            print("halt:", e)

    # --- waiting ---
    def motion_done(self):
        for j, tgt in self.targets.items():
            pos, vel, _ = self._js(self.JOINTS[j])
            if abs(pos - tgt) > self.TOL[j] or abs(vel) > self.STILL.get(j, 0.05):
                return False
        return True

    def wait_motion(self, timeout=15.0):
        self.flush()
        t0, ok = time.time(), 0
        while time.time() - t0 < timeout:
            ok = ok + 1 if self.motion_done() else 0
            if ok >= 4:
                return True
            yield
        now = {j: round(self.joint(j), 3) for j in self.targets}
        raise Abort(f"motion didn't finish in {timeout:.0f} s (targets {self.targets}, now {now}): "
                    "something is in the way or a joint is stuck")

    def wait_gripper(self, timeout=3.0):
        self.flush()
        t0, still = time.time(), 0
        while time.time() - t0 < timeout:
            moving = abs(self._js(self.GRIPPER)[1]) > 0.05
            still = still + 1 if (time.time() - t0 > 0.3 and not moving) else 0
            if still >= 5:
                return True
            yield
        return False

    def wait(self, seconds):
        self.flush()
        t0 = time.time()
        while time.time() - t0 < seconds:
            yield

    def sample(self, t):
        if self.log:
            self.log.writerow([f"{t:.2f}", self.phase, f"{self.joint('lift'):.4f}", f"{self.joint('arm'):.4f}",
                               f"{self.joint('wrist_yaw'):.3f}", f"{self.finger():.3f}",
                               f"{self._js('wrist_extension')[2]:.2f}"])


def make_node():
    import hello_helpers.hello_misc as hm
    from std_msgs.msg import Bool

    node = hm.HelloNode.quick_create("handoff_ros")   # starts rclpy, waits for the driver's action + services
    return node, Bool


def main(argv=None):
    from rclpy.utilities import remove_ros_args
    argv = remove_ros_args(sys.argv if argv is None else ["handoff_ros"] + list(argv))[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_handoff_args(ap)
    ap.add_argument("--gripper-speed", type=float, default=None,
                    help="gripper speed sent to the driver (default: the driver's own default)")
    args = ap.parse_args(argv)
    planned = plan_from_args(ap, args)
    if planned is None:
        return 2
    kin, lim, spot = planned

    import rclpy
    print("connecting to the driver (stretch_driver / sim) ...")
    node, Bool = make_node()
    log_path, log_file, log = new_log(LOG_COLUMNS[:5] + ["finger_rad", "arm_effort_pct"])
    rr = RosRobot(node, spot, args.close_pct, args.holding_pct, args.tug_sign, log, args.gripper_speed)

    def on_runstop(msg):
        rr.runstopped = bool(msg.data)
    node.create_subscription(Bool, "/is_runstopped", on_runstop, 1)
    try:
        t0 = time.time()
        while not node.joint_state.name and time.time() - t0 < 10.0:
            time.sleep(0.1)
        if not node.joint_state.name:
            print("no /stretch/joint_states from the driver after 10 s")
            return 1
        time.sleep(0.5)   # let /is_runstopped arrive
        if rr.runstopped:
            print("run-stop is pressed: release it and try again.")
            return 1
        node.switch_to_position_mode()
        if not args.yes and not confirm():
            return 0
        res = run(rr, kin, lim, args.dry_run)
        return 0 if res.ok else 3
    except Abort as e:
        print("\nABORTED:", e)
        rr.halt()
        return 4
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping.")
        rr.halt()
        return 4
    finally:
        hand_over_if_holding(rr)
        log_file.close()
        print("log:", log_path)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
