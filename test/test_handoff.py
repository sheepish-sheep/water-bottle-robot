"""Tests for the bottle-handoff demo.

    python3 -m pytest test/test_handoff.py -q              # fast: planning math only
    python3 -m pytest test/test_handoff.py -q -m slow      # also full MuJoCo handoffs (~1 min each)
"""
import math
import os
import random

import pytest

from water_bottle_robot.manipulation.handoff_controller import (
    Kinematics, Limits, grasp_height, plan_reach, plan_turn, rotate_into_base)

KIN, LIM = Kinematics(), Limits()


@pytest.mark.parametrize("az_deg", [-170, -120, -90, -45, 0, 30, 90, 150, 179])
@pytest.mark.parametrize("rho", [0.55, 0.7, 0.85])
def test_turn_puts_bottle_on_arm_line(az_deg, rho):
    p = (rho * math.cos(math.radians(az_deg)), rho * math.sin(math.radians(az_deg)), 0.9)
    turn = plan_turn(p[:2], KIN)
    x, y, _ = rotate_into_base(p, turn)
    assert x == pytest.approx(KIN.gx, abs=1e-9)   # on the arm's line
    assert y < 0                                  # on the arm side (robot's right)
    assert abs(turn) <= math.pi + 1e-9


def test_reach_targets_match_kinematics():
    bottle = (KIN.gx, -0.72, 0.95)
    p = plan_reach(bottle, KIN, LIM)
    assert p.feasible
    # grasp point ends up grasp_depth past the bottle axis, at the bottle's height
    assert KIN.gy0 - p.arm_grasp == pytest.approx(bottle[1] - LIM.grasp_depth)
    assert p.lift + KIN.gz0 == pytest.approx(bottle[2])
    assert p.arm_grasp - p.arm_pregrasp == pytest.approx(LIM.pregrasp_standoff)


@pytest.mark.parametrize("bottle, why", [
    ((KIN.gx, -1.20, 0.9), "too far"),
    ((KIN.gx, -0.46, 0.9), "too close"),
    ((KIN.gx, -0.70, 1.40), "lift range"),
    ((KIN.gx, -0.70, 0.10), "lift range"),
])
def test_unreachable_bottles_are_rejected(bottle, why):
    p = plan_reach(bottle, KIN, LIM)
    assert not p.feasible and why in p.reason


def test_grasp_height_clears_the_hand():
    z, ok, _ = grasp_height(hand_top_z=0.80, bottle_top_z=1.05, lim=LIM)
    assert ok and z == pytest.approx(0.80 + LIM.hand_clearance)
    assert z - LIM.finger_below > 0.80 + 0.02   # fingertips stay 2+ cm above the hand


def test_holding_too_high_is_refused():
    z, ok, why = grasp_height(hand_top_z=0.95, bottle_top_z=1.05, lim=LIM)
    assert not ok and "lower" in why


def test_bottle_on_top_of_base_is_rejected():
    assert plan_turn((0.0, 0.0), KIN) is None


@pytest.mark.slow
def test_refuses_when_held_too_high():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from water_bottle_robot.manipulation import handoff_sim as hs
    from water_bottle_robot.manipulation.handoff_scene import BottleSpec
    sc = hs.Scenario(bottle=BottleSpec(body_height=0.17), hold_height=0.10)
    r = hs.run_trial(sc)
    assert "lower" in r["reason"]
    assert r["checks"]["never_touched_person"] and r["checks"]["no_drop"]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [101, 202, 303])
def test_full_handoff_in_sim(seed):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from water_bottle_robot.manipulation import handoff_sim as hs
    sc = hs.random_scenario(random.Random(seed), seed)
    r = hs.run_trial(sc)
    assert r["passed"], (r["checks"], r["reason"], r["scenario"], r["bottle"])
