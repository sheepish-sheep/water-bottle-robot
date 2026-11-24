"""
Bottle handoff behavior for Stretch: take a bottle a person is holding out,
hold it upright, and hand it back. Pure logic -- no MuJoCo or ROS in here.

It talks to the robot only through the small interface below, using motions
the real Stretch supports (rotate base, lift, arm extension, wrist yaw/pitch/
roll, gripper), with speed limits within the real robot's capabilities. The
sim implementation is in handoff_sim.py; a real-robot one would wrap
stretch_body or the stretch_core ROS action server.

Coordinates: robot base frame, x forward, y left, z up (m). With the wrist
straight (yaw=pitch=roll=0), Stretch's grasp point is
    (gx,  gy0 - arm,  lift + gz0)
i.e. the arm reaches out to the robot's RIGHT (-y). The constants come from
the robot model (Kinematics).

Sequence (the FSM in run_handoff):
  LOCATE     coarse bottle estimate (camera)
  FACE       rotate the base so the bottle is on the arm's line
  REFINE     second, closer estimate after turning; grasp height = hand top +
             clearance (refuse if the person holds the bottle too high)
  RAISE      lift (no higher than unfold_lift_max yet), arm retracted
  UNFOLD     wrist straight out, gripper open; then lift the rest of the way
  REACH      extend to a pre-grasp point, then slowly to the grasp point
  GRASP      close; success = fingers stalled on the bottle, not fully closed
  TAKE       hold still and wait for the person to let go
  RETRACT    pull the arm in, bottle upright beside the robot (carry pose)
  GIVE       extend to the hand-back point, wait for the person to grip it and
             give a gentle tug (arm effort changes) -- only then let go
  RELEASE    open, back off slowly
  STOW       retract, fold the wrist, lower
"""
import math
import statistics
from dataclasses import dataclass, field


# ---------- geometry (testable without a robot) ----------

@dataclass
class Kinematics:
    gx: float = -0.021    # grasp point x in base frame (m), wrist straight
    gy0: float = -0.415   # grasp point y at arm = 0
    gz0: float = 0.113    # grasp point z minus lift joint value
    arm_min: float = 0.0
    arm_max: float = 0.52
    lift_min: float = 0.0
    lift_max: float = 1.10


@dataclass
class Limits:
    """Speeds (well within the real robot's limits) and safety margins."""
    lift_speed: float = 0.10        # m/s
    arm_speed: float = 0.12         # m/s, free space
    arm_approach_speed: float = 0.04  # m/s, last few cm to the bottle
    wrist_speed: float = 1.0        # rad/s
    base_turn_speed: float = 0.5    # rad/s
    arm_headroom: float = 0.04      # m: don't plan within this of full extension (sim arm
                                    # stalled at 0.483 of a commanded 0.50; 0.52 is the limit)
    pregrasp_standoff: float = 0.08  # m: stop this far short of the bottle before the slow approach
    grasp_depth: float = 0.05       # m: put the bottle axis this far past the grasp point (toward the palm).
                                    # Shallower (<3 cm) the fingertips pinch the bottle's NEAR side and
                                    # squeeze it out (it pivots, up to ~20 deg). 3.5-7 cm the fingers wrap
                                    # its far side and pull it into the palm (<6 deg, sim sweep); 5 cm is
                                    # the middle of that window, leaving +-1.5 cm for vision error.
    # Unfolding the gripper (wrist yaw pi -> 0) at arm = 0 sweeps the fingertips out to
    # tip_extent past the grasp point, so the bottle's near side must be beyond that.
    tip_extent: float = 0.035       # m, measured from the model (open fingertips, arm 0)
    bottle_radius_max: float = 0.040  # m, largest bottle we plan for
    unfold_margin: float = 0.03     # m

    @property
    def arm_reach_min(self):
        """Smallest arm extension at the grasp, so unfolding can't hit the bottle."""
        return self.grasp_depth + self.bottle_radius_max + self.tip_extent + self.unfold_margin
    take_lift: float = 0.0          # m: raise after closing. 0: lifting while the person still
                                    # grips the bottom twists the bottle in the fingers (+5 deg tilt in sim)
    release_wait: float = 1.8       # s: time for the person to let go
    give_back_offset: float = 0.0   # m: hand back where we took it
    # Grasp height comes from the person's hand: the fingertips reach finger_below
    # under the grasp point, so grasp hand_clearance above the top of the hand.
    finger_below: float = 0.047     # m, measured in sim (fingertip below grasp height)
    hand_clearance: float = 0.07    # m above the hand top = finger_below + 2.3 cm margin
    top_margin: float = 0.08        # m below the bottle's top (cap + neck + shoulder)
    unfold_lift_max: float = 0.75   # m: unfold no higher than this (above it, the unfolding
                                    # gripper can hit the robot's own head)
    handback_timeout: float = 10.0  # s: if nobody tugs on the bottle by then, keep it (don't drop it)
    tug_force: float = 2.5          # N: smallest effort drop that can ever count as a pull (a floor).
                                    # The actual threshold is set each time from the measured noise.
    tug_quiet: float = 0.4          # N std dev: the arm counts as settled below this
    tug_hold: float = 0.25          # s the pull must last (the detector averages over this)
    tug_noise_window: float = 0.5   # s of extra still effort measured after settling (plus the 0.5 s settle
                                    # window) to set the threshold. Short: the person grabs the bottle soon after
    tug_noise_k: float = 5.0        # threshold = this x the noise of the averaged effort (or tug_force if larger).
                                    # 5: over a 10 s wait (~200 windows) jitter alone basically never gets there


@dataclass
class Plan:
    turn: float           # rad, base rotation (+ = counter-clockwise)
    lift: float
    arm_pregrasp: float
    arm_grasp: float
    feasible: bool
    reason: str = ""


def plan_turn(bottle_xy, kin: Kinematics):
    """
    Base rotation that puts the bottle on the arm's line (x = gx, on the -y side).
    Rotating by theta maps a point at polar (rho, alpha) to angle alpha - theta;
    we need rho*cos(alpha - theta) = gx with sin(alpha - theta) < 0.
    """
    bx, by = bottle_xy
    rho = math.hypot(bx, by)
    if rho <= abs(kin.gx) + 1e-6:
        return None
    alpha = math.atan2(by, bx)
    target = -math.acos(kin.gx / rho)   # angle of the arm line point at distance rho (y < 0)
    turn = alpha - target
    return (turn + math.pi) % (2 * math.pi) - math.pi


def grasp_height(hand_top_z, bottle_top_z, lim: Limits):
    """Where to grasp the bottle: clear of the person's hand, below the bottle's shoulder.
    Returns (z, ok, reason)."""
    z = hand_top_z + lim.hand_clearance
    if z > bottle_top_z - lim.top_margin:
        return z, False, "person is holding the bottle too high -- ask them to hold it lower"
    return z, True, ""


def plan_reach(bottle_base, kin: Kinematics, lim: Limits):
    """Lift and arm targets for a grasp point already on the arm line (base frame)."""
    _, by, bz = bottle_base
    arm_grasp = kin.gy0 - by + lim.grasp_depth
    lift = bz - kin.gz0
    p = Plan(0.0, lift, max(kin.arm_min, arm_grasp - lim.pregrasp_standoff), arm_grasp, True)
    if not (kin.lift_min + 0.05 <= lift <= kin.lift_max - 0.02):
        p.feasible, p.reason = False, f"grasp height {bz:.2f} m out of lift range"
    elif arm_grasp > kin.arm_max - lim.arm_headroom:
        p.feasible, p.reason = False, f"bottle too far (needs arm {arm_grasp:.2f} m) -- ask the person to come closer"
    elif arm_grasp < lim.arm_reach_min:
        p.feasible, p.reason = False, f"bottle too close to the robot (arm {arm_grasp:.2f} m)"
    return p


def rotate_into_base(point, turn):
    """Where a base-frame point ends up after the base rotates by `turn`."""
    x, y, z = point
    c, s = math.cos(-turn), math.sin(-turn)
    return (c * x - s * y, s * x + c * y, z)


# ---------- robot interface ----------

class RobotInterface:
    """What the behavior needs from a robot. Calls return immediately; use the wait_* generators."""

    def move(self, joint, target, speed):  # joints: lift, arm, wrist_yaw, wrist_pitch, wrist_roll
        raise NotImplementedError

    def turn_base(self, angle, speed):
        raise NotImplementedError

    def set_gripper(self, opening):  # "open" | "closed"
        raise NotImplementedError

    def joint(self, name):
        raise NotImplementedError

    def arm_effort(self):
        """Arm motor effort/force (N). The real robot reports this as motor effort."""
        raise NotImplementedError

    def gripper_holding(self):
        """True if the fingers are stalled on an object (closed command, not fully closed)."""
        raise NotImplementedError

    def estimate_bottle(self):
        """Perception, in the current base frame, or None if not seen:
        {"axis_xy": (x, y) of the bottle's axis, "top_z": top of the bottle,
         "hand_top_z": top of the hand holding it}. On the real robot: YOLO box for
        the bottle, MediaPipe landmarks for the hand, depth from the hand (opaque),
        not from the clear bottle."""
        raise NotImplementedError

    def wait_motion(self, timeout=15.0):
        """Generator: yields until all commanded motions finish; returns False on timeout."""
        raise NotImplementedError

    def wait_gripper(self, timeout=3.0):
        """Generator: yields until the gripper stops moving (closed on something, or fully
        open/closed); returns False on timeout. Ignores the arm/wrist, which a person
        holding the other end of the bottle will nudge."""
        raise NotImplementedError

    def wait(self, seconds):
        raise NotImplementedError


@dataclass
class Result:
    ok: bool = False
    phase: str = "START"
    reason: str = ""
    plan: Plan = None
    log: list = field(default_factory=list)
    timeouts: int = 0


def run_handoff(robot: RobotInterface, kin=Kinematics(), lim=Limits(), result: Result = None, give_back=True,
                dry_run=False, stow=True):
    """Generator implementing the handoff FSM. `result` is filled in as it runs.

    dry_run: reach to the grasp point with the gripper open, hold there a few
    seconds, then retract and stow without grasping. For lining up the bottle
    spot on the real robot.

    stow: at the end, fold the gripper back next to the mast and lower the lift.
    False leaves the arm retracted with the gripper still pointing out. Folding
    is never done with something in the gripper (a bottle would slip out)."""
    res = result if result is not None else Result()

    def phase(name, note=""):
        res.phase = name
        res.log.append((name, note))

    def settle(timeout=15.0):
        ok = yield from robot.wait_motion(timeout)
        if not ok:
            res.log.append(("TIMEOUT", res.phase))
            res.timeouts += 1
        return ok

    def fail(why):
        res.ok, res.reason = False, why
        res.log.append(("FAIL", why))

    def grip(opening):
        robot.set_gripper(opening)
        ok = yield from robot.wait_gripper()
        if not ok:
            res.log.append(("TIMEOUT", res.phase + " gripper"))
            res.timeouts += 1
        return ok

    def fold():
        if not stow:
            res.log.append(("NO_FOLD", "arm retracted; gripper left pointing out"))
            return
        yield from grip("closed")    # close before folding back next to the mast
        if robot.gripper_holding():  # e.g. the person didn't take the bottle after all
            res.log.append(("NO_FOLD", "something is in the gripper; not folding"))
            return
        robot.move("lift", min(robot.joint("lift"), lim.unfold_lift_max), lim.lift_speed)
        yield from settle()
        robot.move("wrist_yaw", math.pi, lim.wrist_speed)
        robot.move("wrist_pitch", -0.4, lim.wrist_speed)
        yield from settle()
        robot.move("lift", 0.3, lim.lift_speed)
        yield from settle()

    # LOCATE + FACE
    phase("LOCATE")
    est = robot.estimate_bottle()
    if est is None:
        return fail("no bottle detected")
    turn = plan_turn(est["axis_xy"], kin)
    if turn is None:
        return fail("bottle too close to the base")
    phase("FACE", f"turn {math.degrees(turn):+.0f} deg")
    robot.turn_base(turn, lim.base_turn_speed)
    yield from settle()

    # REFINE: re-estimate from the new viewpoint, plan lift/arm
    phase("REFINE")
    est = robot.estimate_bottle()
    if est is None:
        return fail("lost the bottle after turning")
    gz, ok, why = grasp_height(est["hand_top_z"], est["top_z"], lim)
    if not ok:
        return fail(why)
    p = plan_reach((*est["axis_xy"], gz), kin, lim)
    p.turn = turn
    res.plan = p
    if not p.feasible:
        return fail(p.reason)

    # RAISE with the arm in and the gripper tucked, then UNFOLD
    phase("RAISE", f"lift {p.lift:.2f}")
    robot.move("arm", 0.0, lim.arm_speed)
    robot.move("lift", min(p.lift, lim.unfold_lift_max), lim.lift_speed)
    yield from settle()
    phase("UNFOLD")
    robot.move("wrist_pitch", 0.0, lim.wrist_speed)
    robot.move("wrist_roll", 0.0, lim.wrist_speed)
    robot.move("wrist_yaw", 0.0, lim.wrist_speed)
    yield from settle()
    yield from grip("open")          # only once the gripper is out: opening it while
                                     # tucked next to the mast hits the mast
    if p.lift > lim.unfold_lift_max:
        robot.move("lift", p.lift, lim.lift_speed)
        yield from settle()

    # REACH: fast to pre-grasp, slow the last few cm
    phase("REACH", f"arm {p.arm_pregrasp:.2f} -> {p.arm_grasp:.2f}")
    robot.move("arm", p.arm_pregrasp, lim.arm_speed)
    yield from settle()
    robot.move("arm", p.arm_grasp, lim.arm_approach_speed)
    yield from settle()

    if dry_run:
        phase("DRY_RUN", "at the grasp point with the gripper open; not grasping")
        yield from robot.wait(4.0)
        phase("STOW")
        robot.move("arm", 0.0, lim.arm_speed)
        yield from settle()
        yield from fold()
        phase("DONE")
        res.ok = True
        return

    # GRASP
    phase("GRASP")
    yield from grip("closed")
    yield from robot.wait(0.4)
    res.log.append(("CLOSED", "gripper closed and settled; holding" if robot.gripper_holding()
                    else "gripper closed and settled; NOT holding"))
    if not robot.gripper_holding():
        yield from grip("open")
        robot.move("arm", 0.0, lim.arm_speed)
        yield from settle()
        return fail("gripper closed on nothing")

    # TAKE: small lift to take the weight, then wait for the person to let go
    phase("TAKE")
    if lim.take_lift:
        robot.move("lift", p.lift + lim.take_lift, 0.03)
        yield from settle()
    yield from robot.wait(lim.release_wait)
    if not robot.gripper_holding():
        return fail("bottle slipped while taking it")

    # RETRACT to the carry pose
    phase("RETRACT")
    robot.move("arm", 0.0, lim.arm_speed)
    yield from settle()
    if not robot.gripper_holding():
        return fail("dropped the bottle while retracting")
    phase("HOLD")
    yield from robot.wait(1.5)

    if give_back:
        # GIVE: extend back toward the person, then wait for them to grip and tug
        phase("GIVE")
        robot.move("arm", p.arm_grasp - lim.give_back_offset, lim.arm_speed)
        yield from settle()
        # Arm the tug detector only once the arm has settled: with a heavy bottle it keeps
        # swaying for a while after stopping, and a fixed threshold mistook that for a
        # tug (sim: released a 0.7 kg bottle before the person had it).
        win, samples, t = max(1, int(0.5 / robot.dt)), [], 0.0
        while t < 4.0:
            samples = (samples + [robot.arm_effort()])[-win:]
            if len(samples) == win and statistics.pstdev(samples) < lim.tug_quiet:
                break
            yield
            t += robot.dt
        # Measure the noise the detector will actually see. It compares the AVERAGE
        # effort over tug_hold (a pull has to last that long anyway) to the baseline,
        # so the threshold comes from how much that average jitters while nobody
        # touches the bottle -- no hand-tuned number needed (tug_force is only a floor).
        w = max(1, round(lim.tug_hold / robot.dt))
        quiet, t = list(samples), 0.0
        while t < lim.tug_noise_window:
            quiet.append(robot.arm_effort())
            yield
            t += robot.dt
        baseline = statistics.fmean(quiet)
        means = [statistics.fmean(quiet[i:i + w]) for i in range(len(quiet) - w + 1)]
        # Two estimates of the averaged signal's noise; take the larger. Directly measured
        # from ~1 s it comes out low (few independent windows); raw std / sqrt(w) is right
        # for uncorrelated jitter. Underestimating it means releasing on noise.
        noise = max(statistics.pstdev(means) if len(means) > 1 else 0.0,
                    statistics.pstdev(quiet) / math.sqrt(w))
        # The baseline is itself an estimate (from len(quiet) samples); its error adds to
        # the averaged jitter, so the threshold has to cover both.
        noise = math.hypot(noise, statistics.pstdev(quiet) / math.sqrt(len(quiet)))
        threshold = max(lim.tug_force, lim.tug_noise_k * noise)
        res.log.append(("TUG_ARMED", f"baseline {baseline:.2f}, noise {noise:.2f}, threshold {threshold:.2f} "
                                     "(N in sim, effort_pct on the robot)"))
        recent, t = [], 0.0
        while t < lim.handback_timeout:
            # a pull AWAY from the robot makes the arm motor push back: effort drops
            recent = (recent + [robot.arm_effort()])[-w:]
            if len(recent) == w and baseline - statistics.fmean(recent) > threshold:
                break
            yield
            t += robot.dt
        else:
            phase("KEEP", "nobody took the bottle")
            robot.move("arm", 0.0, lim.arm_speed)
            yield from settle()
            return fail("nobody took the bottle back (kept it)")
        # RELEASE: open, back off slowly at first
        phase("RELEASE")
        yield from grip("open")
        yield from robot.wait(0.3)
        robot.move("arm", max(0.0, p.arm_grasp - 0.10), lim.arm_approach_speed)
        yield from settle()

    # STOW
    phase("STOW")
    robot.move("arm", 0.0, lim.arm_speed)
    yield from settle()
    if give_back:
        yield from fold()
    phase("DONE")
    res.ok = True
