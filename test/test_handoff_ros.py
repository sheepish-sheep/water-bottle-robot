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
        self.base_x = 0.0   # how far the base has driven (translate_mobile_base goals)

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
                if k in self.pos:
                    self.pos[k] = v[0] if isinstance(v, tuple) else v
        if accepted and "translate_mobile_base" in pose:
            v = pose["translate_mobile_base"]
            self.base_x += v[0] if isinstance(v, tuple) else v
        handle = types.SimpleNamespace(accepted=accepted,
                                       get_result_async=lambda: types.SimpleNamespace(done=lambda: True))
        fut = types.SimpleNamespace(done=lambda: True, result=lambda: handle)
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


# ---------- gripper camera ALIGN (handoff_ros --camera) ----------

class FakeCamera:
    """gripper_bottle_detector + TF: the bottle at a fixed spot in the world, seen in the
    current base frame (shifted by however far the base has driven), a new estimate per call."""

    def __init__(self, node, x, y, z=0.95, visible=True):
        self.node, self.x, self.y, self.z, self.visible, self.seq = node, x, y, z, visible, 0

    def __call__(self):
        if not self.visible:
            return None
        self.seq += 1
        return self.seq, (self.x - self.node.base_x, self.y, self.z)


def align_run(clock, off_x=0.0, off_reach=0.0, visible=True):
    node = FakeNode()
    rr, kin, lim = make(node)          # measured spot: 0.70 m to the right, on the arm's line
    rr.see_bottle = FakeCamera(node, kin.gx + off_x, -0.70 - off_reach, visible=visible)
    res = handoff_real.run(rr, kin, lim, dry_run=True, align=True)
    planned = handoff_real.plan_reach((kin.gx, -0.70, 0.90), kin, lim).arm_grasp
    arm_cmds = [g["wrist_extension"][0] for g in node.goals if "wrist_extension" in g]
    return res, node, planned, arm_cmds


def test_align_corrects_the_base_and_the_reach(clock):
    res, node, planned, arm = align_run(clock, off_x=0.04, off_reach=0.03)
    assert res.ok, res.log
    assert node.base_x == pytest.approx(0.04)                 # drove forward onto the bottle's line
    assert max(arm) == pytest.approx(planned + 0.03)          # reached 3 cm farther
    assert "ALIGNED" in [n for n, _ in res.log]


def test_align_on_the_spot_doesnt_move_the_base(clock):
    res, node, planned, arm = align_run(clock, off_x=0.005)
    assert res.ok and node.base_x == 0.0
    assert not any("translate_mobile_base" in g for g in node.goals)


def test_align_refuses_a_bottle_far_from_the_spot(clock):
    res, node, planned, arm = align_run(clock, off_x=0.15)
    assert not res.ok and "measured spot" in res.reason
    assert node.base_x == 0.0 and node.pos["wrist_extension"] == pytest.approx(0.0)   # retracted, no chase
    assert max(arm) < planned                                 # never went to the grasp


def test_align_without_a_camera_estimate_fails_safely(clock):
    res, node, planned, arm = align_run(clock, visible=False)
    assert not res.ok and "doesn't see" in res.reason
    assert node.pos["wrist_extension"] == pytest.approx(0.0) and node.base_x == 0.0
