"""Tests for manipulation/handoff_real.py against a FAKE stretch_body.

These check the program's logic (state machine wiring, safety stops, the
"still holding the bottle" exit, argument checks) without a robot. They do not
show the real robot behaves like the fake: that's what the staged tests in
manipulation/REAL_ROBOT.md are for.

    python3 -m pytest test/test_handoff_real.py -q
"""
import os
import sys
import time as real_time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manipulation"))
import handoff_real  # noqa: E402


class Clock:
    """Fake time: advances only when the program sleeps."""
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    strftime = staticmethod(real_time.strftime)


class Joint:
    def __init__(self, clock, pos, speed):
        self.clock, self.pos, self.target, self.speed, self.t = clock, pos, pos, speed, clock.t
        self.vel = 0.0
        self.extra = {}

    def _update(self):
        dt, self.t = self.clock.t - self.t, self.clock.t
        err = self.target - self.pos
        step = min(abs(err), self.speed * dt)
        self.pos += step if err > 0 else -step
        self.vel = (self.speed if step > 0 and abs(self.target - self.pos) > 1e-9 else 0.0)

    @property
    def status(self):
        self._update()
        return {"pos": self.pos, "vel": self.vel, **self.extra}

    def move_to(self, x, v=None):
        self._update()
        self.target = x
        if v:
            self.speed = v


class Prismatic(Joint):
    def __init__(self, clock, pos, effort):
        super().__init__(clock, pos, 0.1)
        self.motor = types.SimpleNamespace()
        self._effort = effort
        self.commands = []

    def move_to(self, x_m, v_m=None):
        self.commands.append(x_m)
        super().move_to(x_m, v_m)

    def move_by(self, dx):
        self.target = self.status["pos"] + dx

    @property
    def motor(self):
        return types.SimpleNamespace(status={"effort_pct": self._effort(self.clock.t)})

    @motor.setter
    def motor(self, _):
        pass


class Gripper(Joint):
    pct_max_open = 50.0

    def __init__(self, clock, bottle):
        super().__init__(clock, 0.0, 200.0)
        self.bottle = bottle   # callable: is a bottle between the fingers?

    @property
    def status(self):
        st = super().status
        st["pos_pct"] = st["pos"]
        return st

    def move_to(self, pct, v=None):
        if pct < 0 and self.bottle():
            pct = 25.0   # fingers stall on a ~7 cm bottle
        elif pct < 0:
            pct = -15.0  # closed on nothing
        super().move_to(pct)


class EndOfArm:
    def __init__(self, clock, bottle):
        self.j = {n: Joint(clock, 3.14 if n == "wrist_yaw" else 0.0, 1.0)
                  for n in ("wrist_yaw", "wrist_pitch", "wrist_roll")}
        self.j["stretch_gripper"] = Gripper(clock, bottle)
        self.joints = list(self.j)

    def get_joint(self, n):
        return self.j[n]

    def move_to(self, n, x, v_r=None):
        self.j[n].move_to(x, v_r)

    def move_by(self, n, dx):
        self.j[n].target = self.j[n].status["pos"] + dx


class FakeRobot:
    def __init__(self, clock, effort=lambda t: 0.0, runstop_at=None, homed=True, bottle=True):
        self.clock = clock
        self.lift = Prismatic(clock, 0.2, lambda t: 0.0)
        self.arm = Prismatic(clock, 0.0, effort)
        # the bottle is between the fingers only while the arm is out at the spot
        self.end_of_arm = EndOfArm(clock, lambda: bottle and self.arm.pos > 0.2)
        self._runstop_at, self._homed = runstop_at, homed
        self.pushes = self.stopped = 0

    @property
    def pimu(self):
        on = self._runstop_at is not None and self.clock.t >= self._runstop_at
        return types.SimpleNamespace(status={"runstop_event": on, "voltage": 12.5})

    def startup(self):
        return True

    def is_homed(self):
        return self._homed

    def push_command(self):
        self.pushes += 1

    def stop(self):
        self.stopped += 1


@pytest.fixture
def env(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(handoff_real, "time", clock)
    monkeypatch.setattr(handoff_real, "LOG_DIR", str(tmp_path))
    holder = {}

    def install(**kw):
        robot = FakeRobot(clock, **kw)
        holder["robot"] = robot
        mod = types.ModuleType("stretch_body.robot")
        mod.Robot = lambda: robot
        pkg = types.ModuleType("stretch_body")
        pkg.robot = mod
        monkeypatch.setitem(sys.modules, "stretch_body", pkg)
        monkeypatch.setitem(sys.modules, "stretch_body.robot", mod)
        return robot
    return clock, install, tmp_path


SPOT = ["--distance", "0.70", "--hand-top", "0.83"]


def test_dry_run_reaches_and_comes_back_without_grasping(env):
    clock, install, logs = env
    robot = install()
    assert handoff_real.main(SPOT + ["--dry-run", "--yes"]) == 0
    p = handoff_real.plan_reach((0, -0.70, 0.90), handoff_real.Kinematics(), handoff_real.Limits())
    assert max(robot.arm.commands) == pytest.approx(p.arm_grasp)   # went to the grasp point
    assert robot.arm.status["pos"] == pytest.approx(0.0)          # and back
    assert robot.lift.status["pos"] == pytest.approx(p.lift)      # not folded/lowered by default
    assert robot.end_of_arm.get_joint("wrist_yaw").status["pos"] == pytest.approx(0.0)   # gripper still out
    assert robot.stopped == 1
    assert len(os.listdir(logs)) == 1


def test_full_handoff_releases_on_tug(env):
    clock, install, _ = env
    # the person pulls (effort drops 20 pct) from t = 46 s on: GIVE arms the tug detector at ~44 s
    robot = install(effort=lambda t: -20.0 if t > 46 else 0.0)
    assert handoff_real.main(SPOT + ["--tug-threshold", "8", "--yes"]) == 0
    g = robot.end_of_arm.get_joint("stretch_gripper")
    assert g.status["pos_pct"] == pytest.approx(50.0)   # opened to hand it back, left open
    assert robot.arm.status["pos"] == pytest.approx(0.0)
    assert robot.end_of_arm.get_joint("wrist_yaw").status["pos"] == pytest.approx(0.0)   # no fold


def test_stow_option_folds_after_handing_back(env):
    clock, install, _ = env
    robot = install(effort=lambda t: -20.0 if t > 46 else 0.0)
    assert handoff_real.main(SPOT + ["--tug-threshold", "8", "--stow", "--yes"]) == 0
    assert robot.lift.status["pos"] == pytest.approx(0.3)
    assert robot.end_of_arm.get_joint("wrist_yaw").status["pos"] == pytest.approx(3.14159, abs=1e-3)


def test_never_folds_with_something_in_the_gripper(env, monkeypatch):
    clock, install, _ = env
    # the tug "releases" the bottle, but the fake keeps it between the fingers whenever
    # the arm is out, so closing before the fold finds it still there
    robot = install(effort=lambda t: -20.0 if t > 46 else 0.0)
    robot.end_of_arm.j["stretch_gripper"].bottle = lambda: True
    monkeypatch.setattr("builtins.input", lambda msg="": "")
    assert handoff_real.main(SPOT + ["--tug-threshold", "8", "--stow", "--yes"]) == 0
    assert robot.end_of_arm.get_joint("wrist_yaw").status["pos"] == pytest.approx(0.0)   # didn't fold


def test_nobody_takes_it_back_keeps_it_and_asks_before_letting_go(env, monkeypatch):
    clock, install, _ = env
    robot = install()          # effort never changes: nobody tugs
    asked = []
    monkeypatch.setattr("builtins.input", lambda msg="": asked.append(msg) or "")
    assert handoff_real.main(SPOT + ["--tug-threshold", "8", "--yes"]) == 3
    assert asked and "still holding" in asked[0]
    assert robot.end_of_arm.get_joint("stretch_gripper").status["pos_pct"] == pytest.approx(50.0)  # opened after Enter


def test_runstop_aborts_and_halts(env):
    clock, install, _ = env
    robot = install(runstop_at=5.0)
    assert handoff_real.main(SPOT + ["--dry-run", "--yes"]) == 4
    assert robot.stopped == 1
    assert robot.lift.status["pos"] < 0.75   # stopped partway up


def test_not_homed_refuses_before_moving(env):
    clock, install, _ = env
    robot = install(homed=False)
    assert handoff_real.main(SPOT + ["--dry-run", "--yes"]) == 1
    assert robot.lift.commands == [] and robot.arm.commands == []


def test_closing_on_nothing_fails_safely(env):
    clock, install, _ = env
    robot = install(bottle=False)
    assert handoff_real.main(SPOT + ["--tug-threshold", "8", "--yes"]) == 3
    assert robot.arm.status["pos"] == pytest.approx(0.0)   # pulled back


def test_operator_must_type_go(env, monkeypatch):
    clock, install, _ = env
    robot = install()
    monkeypatch.setattr("builtins.input", lambda msg="": "no")
    assert handoff_real.main(SPOT + ["--dry-run"]) == 0
    assert robot.lift.commands == []


@pytest.mark.parametrize("dist", [0.40, 1.00])
def test_unreachable_spot_is_refused_without_connecting(env, dist):
    clock, install, _ = env
    robot = install()
    assert handoff_real.main(["--distance", str(dist), "--hand-top", "0.83", "--dry-run", "--yes"]) == 2
    assert robot.stopped == 0


def test_handoff_needs_a_measured_tug_threshold(env):
    with pytest.raises(SystemExit):
        handoff_real.main(SPOT + ["--yes"])


def test_tug_threshold_from_a_run_log(env, monkeypatch, capsys):
    import handoff_real_check
    clock, install, logs = env
    # GIVE's arm stops at ~44 s; the person waits 3 s, then pulls (effort rises 12 pct: sign -1)
    install(effort=lambda t: 12.0 if 48 < t < 52 else 0.3 * ((t * 20) % 2))
    monkeypatch.setattr("builtins.input", lambda msg="": "")
    assert handoff_real.main(SPOT + ["--tug-threshold", "1000", "--yes"]) == 3   # kept it, as intended
    (log,) = os.listdir(logs)
    threshold, sign = handoff_real_check.tug_log(os.path.join(logs, log))
    assert sign == -1 and 4 < threshold < 8
