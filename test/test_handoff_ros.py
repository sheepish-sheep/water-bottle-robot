"""Tests for water_bottle_robot/manipulation/handoff_ros.py against a FAKE driver node.

Checks how the ROS adapter talks to stretch_driver: moves batched into one goal,
an explicit non-zero speed on every joint (a missing speed reaches the wrist and
gripper motors as 0 = no speed limit), the gripper in a goal of its own, retries
when the driver rejects a goal, and stopping on the run-stop. Not a substitute
for scripts/run_handoff_ros_sim.sh (real driver code in MuJoCo) or the robot.

    python3 -m pytest test/test_handoff_ros.py -q     # needs ~/ament_ws sourced (hello_helpers)
"""
import types

import pytest

pytest.importorskip("hello_helpers.gripper_conversion")

from water_bottle_robot.manipulation import handoff_real, handoff_ros  # noqa: E402
from water_bottle_robot.manipulation.handoff_controller import Kinematics, Limits  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeNode:
    """Stands in for HelloNode + stretch_driver: goals take effect at once."""

    NAMES = ["joint_lift", "wrist_extension", "joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll",
             "joint_gripper_finger_left"]

    def __init__(self, reject_first=0):
        self.pos = {"joint_lift": 0.2, "wrist_extension": 0.0, "joint_wrist_yaw": 3.14,
                    "joint_wrist_pitch": -0.4, "joint_wrist_roll": 0.0, "joint_gripper_finger_left": 0.0}
        self.goals, self.stopped, self.reject_left = [], 0, reject_first

    @property
    def joint_state(self):
        return types.SimpleNamespace(name=self.NAMES, position=[self.pos[n] for n in self.NAMES],
                                     velocity=[0.0] * len(self.NAMES), effort=[0.0] * len(self.NAMES))

    def move_to_pose(self, pose, blocking=True):
        accepted = self.reject_left == 0
        self.reject_left = max(0, self.reject_left - 1)
        if accepted:
            self.goals.append(dict(pose))
            for k, v in pose.items():
                self.pos[k] = v[0] if isinstance(v, tuple) else v
        fut = types.SimpleNamespace(done=lambda: True,
                                    result=lambda: types.SimpleNamespace(accepted=accepted))
        return fut

    def stop_the_robot(self):
        self.stopped += 1


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(handoff_real, "time", c)
    monkeypatch.setattr(handoff_ros, "time", c)
    return c


def make(node):
    kin, lim = Kinematics(), Limits()
    spot = handoff_real.fixed_spot(0.70, 0.83, kin, lim)
    return handoff_ros.RosRobot(node, spot), kin, lim


def test_dry_run_goals_are_batched_with_explicit_speeds(clock):
    node = FakeNode()
    rr, kin, lim = make(node)
    res = handoff_real.run(rr, kin, lim, dry_run=True)
    assert res.ok, res.log
    assert node.goals, "no goals sent"
    for g in node.goals:
        if "joint_gripper_finger_left" in g:
            assert len(g) == 1                         # gripper alone, driver's default speed
        else:
            assert all(isinstance(v, tuple) and v[1] > 0 for v in g.values()), g
    # arm + lift went out together in one goal (RAISE), the three wrist joints in another (UNFOLD)
    assert {"wrist_extension", "joint_lift"} in [set(g) for g in node.goals]
    assert {"joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"} in [set(g) for g in node.goals]
    p = handoff_real.plan_reach((0, -0.70, 0.90), kin, lim)
    arm_targets = [g["wrist_extension"][0] for g in node.goals if "wrist_extension" in g]
    assert max(arm_targets) == pytest.approx(p.arm_grasp)
    # by default it doesn't fold or lower at the end: arm in, lift where it was, gripper still out
    assert node.pos["wrist_extension"] == pytest.approx(0.0) and node.pos["joint_lift"] == pytest.approx(p.lift)
    assert node.pos["joint_wrist_yaw"] == pytest.approx(0.0)


def test_stow_folds_through_the_driver(clock):
    node = FakeNode()
    rr, kin, lim = make(node)
    assert handoff_real.run(rr, kin, lim, dry_run=True, stow=True).ok
    assert node.pos["joint_lift"] == pytest.approx(0.3)
    assert node.pos["joint_wrist_yaw"] == pytest.approx(3.14159, abs=1e-3)


def test_rejected_goal_is_retried(clock):
    node = FakeNode(reject_first=2)
    rr, kin, lim = make(node)
    assert handoff_real.run(rr, kin, lim, dry_run=True).ok


def test_driver_that_keeps_rejecting_aborts(clock):
    node = FakeNode(reject_first=10 ** 6)
    rr, kin, lim = make(node)
    with pytest.raises(handoff_real.Abort, match="rejected"):
        handoff_real.run(rr, kin, lim, dry_run=True)


def test_runstop_aborts(clock):
    node = FakeNode()
    rr, kin, lim = make(node)
    rr.runstopped = True
    with pytest.raises(handoff_real.Abort, match="run-stop"):
        handoff_real.run(rr, kin, lim, dry_run=True)
    rr.halt()
    assert node.stopped == 1


def test_gripper_targets_use_the_same_pct_units_as_stretch_body(clock):
    from hello_helpers.gripper_conversion import GripperConversion
    rr, _, _ = make(FakeNode())
    conv = GripperConversion()
    assert rr.close_rad == pytest.approx(conv.robotis_to_finger(-80.0))       # default squeeze
    assert rr.holding_rad is None                                            # no position check by default
    assert rr.open_rad > conv.robotis_to_finger(330.0)   # wider than our robot opens: the driver clips it
    rr2 = handoff_ros.RosRobot(FakeNode(), rr.spot, close_pct=-80.0, holding_pct=5.0)
    assert rr2.close_rad < rr2.holding_rad < rr2.open_rad
