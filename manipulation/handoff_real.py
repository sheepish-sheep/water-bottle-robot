"""
Bottle handoff on the REAL Stretch, fixed-spot version (no camera yet).

The person holds the bottle at a spot you measure beforehand: straight out to
the robot's right (the side the arm reaches), DIST meters from the middle of
the base, with the top of their hand HAND_TOP meters above the floor. The
robot runs the same state machine as the simulation (handoff_controller.py):
raise, unfold, reach, grasp, wait for the person to let go, hold, hand back
on a tug, stow.

Runs ON THE ROBOT's computer (stretch_body is preinstalled there). Only one
program can drive the robot at a time: stop stretch_driver / ROS first.

    # 1. line up the spot: reach there with the gripper open, don't grasp
    python3 manipulation/handoff_real.py --distance 0.70 --hand-top 0.83 --dry-run
    # 2. the handoff (tug threshold measured with handoff_real_check.py tug)
    python3 manipulation/handoff_real.py --distance 0.70 --hand-top 0.83 --tug-threshold 8 --tug-sign 1

Safety: it asks before moving, runs at half speed by default (--speed), stops
everything on Ctrl-C or the run-stop button, and stops if any motion doesn't
finish in time. Keep a hand on the run-stop.

Every run writes a log (phases + joint/effort samples) to
~/ament_ws/artifacts/handoff_real/.
"""
import argparse
import csv
import dataclasses
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handoff_controller import Kinematics, Limits, Result, RobotInterface, grasp_height, plan_reach, run_handoff  # noqa: E402

LOG_DIR = os.path.expanduser("~/ament_ws/artifacts/handoff_real")


class Abort(Exception):
    """Stop the run: halt all motion and exit."""


class StretchRobot(RobotInterface):
    """RobotInterface on stretch_body (lift, arm, wrist yaw/pitch/roll, gripper). No base motion."""

    dt = 0.05   # control tick (s)
    TOL = {"lift": 0.01, "arm": 0.01, "wrist_yaw": 0.05, "wrist_pitch": 0.05, "wrist_roll": 0.05}
    STILL = {"lift": 0.005, "arm": 0.005}   # m/s; wrist joints use 0.05 rad/s

    def __init__(self, robot, spot, close_pct=-50.0, holding_pct=5.0, tug_sign=1.0, log=None):
        self.r = robot
        self.spot = spot
        self.close_pct, self.holding_pct, self.tug_sign = close_pct, holding_pct, tug_sign
        self.log = log
        self.targets = {}
        self.grip_cmd = None
        self.phase = ""
        self.eoa = set(robot.end_of_arm.joints)
        for need in ("wrist_yaw", "stretch_gripper"):
            if need not in self.eoa:
                raise Abort(f"this robot has no '{need}' joint (end of arm joints: {sorted(self.eoa)})")
        g = robot.end_of_arm.get_joint("stretch_gripper")
        self.open_pct = float(getattr(g, "pct_max_open", 50.0))

    # --- state ---
    def _gripper(self):
        return self.r.end_of_arm.get_joint("stretch_gripper")

    def _status(self, name):
        if name == "lift":
            return self.r.lift.status
        if name == "arm":
            return self.r.arm.status
        return self.r.end_of_arm.get_joint(name).status

    def joint(self, name):
        return float(self._status(name)["pos"])

    def arm_effort(self):
        return self.tug_sign * float(self.r.arm.motor.status["effort_pct"])

    def gripper_pct(self):
        return float(self._gripper().status["pos_pct"])

    def gripper_holding(self):
        # pct 0 = fingertips just touching; closed on nothing goes below 0,
        # closed on a 6-8 cm bottle stays well above it.
        return self.grip_cmd == "closed" and self.gripper_pct() > self.holding_pct

    def estimate_bottle(self):
        return dict(self.spot)

    def check_safety(self):
        if self.r.pimu.status.get("runstop_event"):
            raise Abort("run-stop pressed")

    # --- commands ---
    def move(self, joint, target, speed):
        if joint == "lift":
            self.r.lift.move_to(target, v_m=speed)
            self.r.push_command()
        elif joint == "arm":
            self.r.arm.move_to(target, v_m=speed)
            self.r.push_command()
        elif joint in self.eoa:
            self.r.end_of_arm.move_to(joint, target, v_r=speed)
        else:
            return   # e.g. no wrist pitch/roll on a robot without the dex wrist
        self.targets[joint] = target

    def turn_base(self, angle, speed):
        # Fixed-spot mode puts the bottle on the arm's line, so the plan never turns.
        if abs(angle) > math.radians(1.0):
            raise Abort(f"plan wants a {math.degrees(angle):.1f} deg base turn; base motion is disabled here")

    def set_gripper(self, opening):
        pct = self.open_pct if opening == "open" else self.close_pct
        self.r.end_of_arm.move_to("stretch_gripper", pct)
        self.grip_cmd = opening

    def halt(self):
        """Stop every joint where it is (same calls stretch_driver uses to stop)."""
        r = self.r
        for f in (lambda: r.arm.move_by(0.0), lambda: r.lift.move_by(0.0)):
            try:
                f()
            except Exception as e:   # keep going: stopping the rest matters more
                print("halt:", e)
        for j in self.eoa:
            try:
                r.end_of_arm.move_by(j, 0.0)
            except Exception as e:
                print("halt:", e)
        try:
            r.push_command()
        except Exception as e:
            print("halt:", e)

    # --- waiting (generators; one yield = one control tick) ---
    def motion_done(self):
        for j, tgt in self.targets.items():
            st = self._status(j)
            if abs(st["pos"] - tgt) > self.TOL[j] or abs(st["vel"]) > self.STILL.get(j, 0.05):
                return False
        return True

    def wait_motion(self, timeout=15.0):
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
        # A person holding the bottle can nudge it, so this only watches the gripper.
        t0, still = time.time(), 0
        while time.time() - t0 < timeout:
            moving = abs(self._gripper().status["vel"]) > 0.05
            still = still + 1 if (time.time() - t0 > 0.3 and not moving) else 0
            if still >= 5:
                return True
            yield
        return False

    def wait(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            yield

    # --- logging ---
    def sample(self, t):
        if self.log:
            self.log.writerow([f"{t:.2f}", self.phase, f"{self.joint('lift'):.4f}", f"{self.joint('arm'):.4f}",
                               f"{self.joint('wrist_yaw'):.3f}", f"{self.gripper_pct():.1f}",
                               f"{self.r.arm.motor.status['effort_pct']:.2f}"])


def fixed_spot(distance, hand_top, kin: Kinematics, lim: Limits):
    """The 'perception' result for a bottle held at a measured spot on the arm's line."""
    grasp_z = hand_top + lim.hand_clearance
    return {"axis_xy": (kin.gx, -distance), "hand_top_z": hand_top,
            # assume the bottle sticks out far enough above the hand (checked by the operator)
            "top_z": grasp_z + lim.top_margin}


def build_limits(args):
    s = args.speed
    lim = Limits()
    lim = dataclasses.replace(
        lim, lift_speed=lim.lift_speed * s, arm_speed=lim.arm_speed * s,
        arm_approach_speed=lim.arm_approach_speed * s, wrist_speed=lim.wrist_speed * s)
    if args.tug_threshold is not None:
        # effort is in effort_pct on the real robot, not N
        lim = dataclasses.replace(lim, tug_force=args.tug_threshold, tug_quiet=args.tug_threshold / 3)
    return lim


def describe(spot, kin, lim):
    z, ok, why = grasp_height(spot["hand_top_z"], spot["top_z"], lim)
    p = plan_reach((*spot["axis_xy"], z), kin, lim)
    return z, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=float, required=True,
                    help="m, sideways from the middle of the base (between the drive wheels) to the bottle's "
                         "center, on the robot's right. Works for about 0.53-0.84")
    ap.add_argument("--hand-top", type=float, required=True,
                    help="m above the floor: top of the person's hand holding the bottle. "
                         "The robot grasps 7 cm above it; at least 15 cm of bottle must stick out above the hand")
    ap.add_argument("--dry-run", action="store_true",
                    help="reach to the grasp point with the gripper open, hold 4 s, come back. No grasp")
    ap.add_argument("--tug-threshold", type=float,
                    help="arm effort change (effort_pct) that counts as the person pulling. "
                         "Measure it with handoff_real_check.py tug. Required unless --dry-run")
    ap.add_argument("--tug-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="+1 if pulling the gripper away from the robot makes arm effort go DOWN, -1 if UP "
                         "(handoff_real_check.py tug tells you)")
    ap.add_argument("--speed", type=float, default=0.5, help="fraction of the sim speeds (default 0.5)")
    ap.add_argument("--close-pct", type=float, default=-50.0,
                    help="gripper close target (stretch_body pct; -100 = hardest squeeze). Default -50")
    ap.add_argument("--holding-pct", type=float, default=5.0,
                    help="gripper pct above which 'closed' means 'holding something' (0 = fingertips touch)")
    ap.add_argument("--yes", action="store_true", help="don't ask before moving")
    args = ap.parse_args(argv)

    if not 0.1 <= args.speed <= 1.0:
        ap.error("--speed must be between 0.1 and 1.0")
    if not args.dry_run and args.tug_threshold is None:
        ap.error("--tug-threshold is required for a real handoff (measure it: handoff_real_check.py tug)")

    kin, lim = Kinematics(), build_limits(args)
    spot = fixed_spot(args.distance, args.hand_top, kin, lim)
    z, plan = describe(spot, kin, lim)
    print(f"bottle spot: {args.distance:.2f} m to the robot's right, hand top {args.hand_top:.2f} m")
    print(f"plan: grasp at {z:.2f} m (lift {plan.lift:.2f}), arm {plan.arm_pregrasp:.2f} -> {plan.arm_grasp:.2f} m, "
          f"speed x{args.speed}, {'DRY RUN (no grasp)' if args.dry_run else 'full handoff'}")
    if not plan.feasible:
        print("can't do this spot:", plan.reason)
        return 2

    import stretch_body.robot   # only on the robot
    robot = stretch_body.robot.Robot()
    if not robot.startup():
        print("couldn't connect to the robot. Is stretch_driver / another program using it? "
              "Stop it (or run stretch_free_robot_process.py) and try again.")
        return 1
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, time.strftime("run_%Y%m%d_%H%M%S.csv"))
    sr = None
    try:
        if not robot.is_homed():
            print("robot isn't homed: run stretch_robot_home.py first (it moves the robot).")
            return 1
        if robot.pimu.status.get("runstop_event"):
            print("run-stop is pressed: release it (hold the button ~1 s) and try again.")
            return 1
        with open(log_path, "w", newline="") as f:
            log = csv.writer(f)
            log.writerow(["t", "phase", "lift", "arm", "wrist_yaw", "gripper_pct", "arm_effort_pct"])
            sr = StretchRobot(robot, spot, args.close_pct, args.holding_pct, args.tug_sign, log)
            if not args.yes:
                print("\nThe robot will raise its lift, unfold the gripper and reach out to its RIGHT.")
                print("Clear the area, keep a hand on the run-stop.")
                if input("Type 'go' to start: ").strip().lower() != "go":
                    print("cancelled; nothing moved.")
                    return 0
            res = Result()
            gen = run_handoff(sr, kin, lim, res, give_back=True, dry_run=args.dry_run)
            t0, shown = time.time(), 0
            for _ in gen:
                sr.check_safety()
                while shown < len(res.log):
                    name, note = res.log[shown]
                    print(f"[{time.time() - t0:6.1f}s] {name} {note}  (gripper {sr.gripper_pct():.1f} pct, holding if > {sr.holding_pct:g})", flush=True)
                    shown += 1
                sr.phase = res.phase
                sr.sample(time.time() - t0)
                time.sleep(sr.dt)
            for name, note in res.log[shown:]:
                print(f"[{time.time() - t0:6.1f}s] {name} {note}  (gripper {sr.gripper_pct():.1f} pct, holding if > {sr.holding_pct:g})", flush=True)
            print("\nRESULT:", "OK" if res.ok else f"FAILED in {res.phase}: {res.reason}")
            return 0 if res.ok else 3
    except Abort as e:
        print("\nABORTED:", e)
        if sr:
            sr.halt()
        return 4
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping.")
        if sr:
            sr.halt()
        return 4
    finally:
        if sr and sr.gripper_holding():
            # shutting down can let the gripper go limp: don't drop the bottle
            try:
                input("The robot is still holding the bottle. Take hold of it and press Enter to open the gripper.")
                sr.set_gripper("open")
                time.sleep(1.5)
            except (EOFError, KeyboardInterrupt):
                pass
        print("log:", log_path)
        robot.stop()


if __name__ == "__main__":
    sys.exit(main())
