"""
Read-only checks on the real Stretch before the handoff. NOTHING HERE MOVES
THE ROBOT: it only reads status. Run on the robot's computer, with
stretch_driver / ROS stopped (only one program can talk to the robot).

    python3 manipulation/handoff_real_check.py status   # homed? run-stop? battery, joints, gripper
    python3 manipulation/handoff_real_check.py watch    # live joints + arm effort + gripper (Ctrl-C to stop)
    python3 manipulation/handoff_real_check.py tug      # measure the tug threshold for handoff_real.py
    python3 manipulation/handoff_real_check.py tug-log RUN.csv   # same, from a handoff_real.py run's log

`tug`: with the arm out a bit and the gripper unfolded (put it there with
the robot's usual tools, e.g. the Xbox controller / stretch_xbox_controller_teleop.py,
then quit that), it records the arm motor effort while nobody touches the
robot, then while someone pulls the gripper gently away from the robot (the
direction the arm extends) a few times. It prints the threshold the automatic
tug detector would use, whether your pull would release the bottle, and the
--tug-sign to use.

`tug-log`: if `tug` shows almost no change (the arm may not hold position
when no program is commanding it), measure during a real handoff instead:
run handoff_real.py with --tug-threshold 1000 (a floor so high it never releases), and when it
offers the bottle back (GIVE), wait 3 s, then pull gently a few times. It
keeps the bottle and asks you to take it. Then run tug-log on the log it
printed. This needs no robot connection.
"""
import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handoff_controller import Limits  # noqa: E402


def connect():
    import stretch_body.robot   # only on the robot
    r = stretch_body.robot.Robot()
    if not r.startup():
        print("couldn't connect. Is stretch_driver / another program using the robot? Stop it first.")
        sys.exit(1)
    time.sleep(0.5)   # let the status threads fill in
    return r


def joints(r):
    out = {"lift": r.lift.status["pos"], "arm": r.arm.status["pos"]}
    for j in r.end_of_arm.joints:
        st = r.end_of_arm.get_joint(j).status
        out[j] = st.get("pos_pct", st["pos"]) if j == "stretch_gripper" else st["pos"]
    return out


def status(r):
    print("homed:     ", r.is_homed(), "" if r.is_homed() else " <- run stretch_robot_home.py (it moves the robot)")
    print("run-stop:  ", "PRESSED" if r.pimu.status.get("runstop_event") else "released")
    v = r.pimu.status.get("voltage", 0.0)
    print(f"battery:    {v:.1f} V", " <- low, charge it" if 0 < v < 11.0 else "")
    print("end of arm:", list(r.end_of_arm.joints))
    for k, v in joints(r).items():
        unit = "pct" if k == "stretch_gripper" else ("m" if k in ("lift", "arm") else "rad")
        print(f"  {k:16s} {v:8.3f} {unit}")
    g = r.end_of_arm.get_joint("stretch_gripper")
    ap = g.status.get("gripper_conversion", {}).get("aperture_m")
    if ap is not None:
        print(f"  gripper aperture {ap * 100:6.1f} cm")
    print(f"  arm effort       {r.arm.motor.status['effort_pct']:8.2f} pct")


def watch(r):
    print("Ctrl-C to stop")
    try:
        while True:
            j = joints(r)
            line = "  ".join(f"{k}={v:.3f}" for k, v in j.items())
            print(f"{line}  arm_effort={r.arm.motor.status['effort_pct']:.2f}", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def record(r, seconds):
    xs, t0 = [], time.time()
    while time.time() - t0 < seconds:
        xs.append(float(r.arm.motor.status["effort_pct"]))
        time.sleep(0.02)
    return xs


def tug(r):
    print(f"arm at {r.arm.status['pos']:.3f} m. Gripper should be unfolded and the arm out ~0.2 m or more.")
    input("1/2  Nobody touch the robot. Press Enter, then wait 5 s ...")
    still = record(r, 5.0)
    input("2/2  Hold the gripper and pull it GENTLY away from the robot (the way the arm extends), "
          "2-3 times, for 6 s. Press Enter and start ...")
    pull = record(r, 6.0)
    suggest(still, pull, dt=0.02)


def suggest(still, pull, dt, floor=3.0):
    """What handoff_real.py's automatic tug detector would do with these effort samples
    (still = nobody touching, pull = someone pulling), sampled every `dt` s.
    Same rule as handoff_controller.py: average over tug_hold, threshold = max(floor,
    tug_noise_k x the noise of that average)."""
    lim = Limits()
    w = max(1, round(lim.tug_hold / dt))

    def avg(xs):
        return [statistics.fmean(xs[i:i + w]) for i in range(len(xs) - w + 1)]
    base = statistics.fmean(still)
    noise = max(statistics.pstdev(avg(still)) if len(still) > w else 0.0,
                statistics.pstdev(still) / w ** 0.5)       # same estimate as the controller
    noise = (noise ** 2 + statistics.pstdev(still) ** 2 / len(still)) ** 0.5
    threshold = max(floor, lim.tug_noise_k * noise)
    pulled = [m - base for m in avg(pull)] or [0.0]
    change = min(pulled) if abs(min(pulled)) > abs(max(pulled)) else max(pulled)
    sign = 1 if change < 0 else -1          # handoff_real.py expects pulls to make (sign * effort) drop
    released = abs(change) > threshold
    print(f"still: mean {base:.2f} pct, noise of the {lim.tug_hold:g} s average {noise:.2f} -> "
          f"automatic threshold {threshold:.2f}")
    print(f"pulls (averaged over {lim.tug_hold:g} s): up to {change:+.2f} pct -> "
          f"{'WOULD release the bottle' if released else 'would NOT release the bottle'}")
    print(f"\n--tug-sign {sign}" + ("" if sign == 1 else "   (not the default: add it to the command)"))
    if not released:
        print("Pull a bit harder or longer (at least 1/4 s), or lower the floor with --tug-threshold "
              f"(currently {floor:g}) if the threshold above is just the floor.")
    return threshold, sign, released


def tug_log(path):
    """From a handoff_real.py log: GIVE phase, after the arm stops; first 2 s = still, rest = pulls."""
    rows = [r for r in csv.DictReader(open(path)) if r["phase"] == "GIVE"]
    if not rows:
        print("no GIVE phase in this log (the run didn't get to the hand-back)")
        return None
    arm_end = float(rows[-1]["arm"])
    # the arm is out at the hand-back point from the first sample within 1 cm of where it ended
    i0 = next(i for i, r in enumerate(rows) if abs(float(r["arm"]) - arm_end) < 0.01)
    t0 = float(rows[i0]["t"])
    eff = [(float(r["t"]) - t0, float(r["arm_effort_pct"])) for r in rows[i0:]]
    still = [e for t, e in eff if 0.5 <= t < 2.5]
    pull = [e for t, e in eff if t >= 3.0]
    if len(still) < 10 or len(pull) < 10:
        print("GIVE phase too short in this log")
        return None
    dt = statistics.median(b[0] - a[0] for a, b in zip(eff, eff[1:]))
    return suggest(still, pull, dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["status", "watch", "tug", "tug-log"], nargs="?", default="status")
    ap.add_argument("log", nargs="?", help="for tug-log: a run_*.csv from handoff_real.py")
    args = ap.parse_args()
    if args.what == "tug-log":
        if not args.log:
            ap.error("tug-log needs the log file")
        tug_log(args.log)
        return
    r = connect()
    try:
        {"status": status, "watch": watch, "tug": tug}[args.what](r)
    finally:
        r.stop()


if __name__ == "__main__":
    main()
