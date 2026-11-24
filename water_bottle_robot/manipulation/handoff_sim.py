#!/usr/bin/env python3
"""
Run the Stretch bottle-handoff demo in MuJoCo.

    ros2 run water_bottle_robot handoff_sim --video ~/ros2_ws/artifacts/handoff.mp4   # render a video (headless)
    ros2 run water_bottle_robot handoff_sim --viewer                                    # watch live (needs a desktop)
    ros2 run water_bottle_robot handoff_sim --trials 50 --seed 1                        # randomized pass/fail batch

The robot logic is handoff_controller.py; this file supplies the MuJoCo robot
(speed-limited joint setpoints), a perception stand-in (true position + noise),
a scripted person, and the checks.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from water_bottle_robot.manipulation.handoff_controller import Kinematics, Limits, Result, RobotInterface, run_handoff
from water_bottle_robot.manipulation.handoff_scene import BottleSpec, build_model, set_weld, yaw_quat

CTRL_DT = 0.01  # s, control/metrics rate


@dataclass
class Scenario:
    bottle_rho: float = 0.68       # m, bottle distance from robot base center
    bottle_azimuth: float = -90.0  # deg, direction from robot (0 = straight ahead, -90 = robot's right)
    grasp_height: float = 0.90     # m, approx. height of the robot's grasp (sets where the person holds the bottle)
    bottle: BottleSpec = field(default_factory=BottleSpec)
    grip_force: float = 30.0       # N cap on the gripper actuator
    vision_noise: float = 0.01     # m, std dev per axis of the bottle estimate
    tremor: float = 0.006          # m, amplitude of the person's hand drift
    release_delay: float = 0.6     # s after the robot grips before the person lets go
    grab_delay: float = 0.6        # s after the robot offers the bottle before the person holds it
    hold_height: float = 0.045     # m above the bottle's bottom where the person's hand is (its center)
    seed: int = 0


def random_scenario(rng: random.Random, seed: int) -> Scenario:
    body_h = rng.uniform(0.17, 0.24)
    return Scenario(
        bottle_rho=rng.uniform(0.54, 0.83),   # reachable band: arm 0.155..0.48 m
        bottle_azimuth=rng.uniform(-180, 180),
        grasp_height=rng.uniform(0.65, 1.10),
        bottle=BottleSpec(radius=rng.uniform(0.032, 0.040), body_height=body_h,
                          mass=rng.uniform(0.03, 0.80), friction=rng.uniform(0.4, 1.0)),
        grip_force=rng.uniform(15.0, 40.0),
        vision_noise=0.01,
        tremor=rng.uniform(0.0, 0.010),
        release_delay=rng.uniform(0.2, 1.2),
        grab_delay=rng.uniform(0.3, 1.0),
        # where the person holds it: low enough that there's room to grasp above the hand
        # (grasp lands at hold + 0.115; must stay below body_h - 0.01 even with vision noise)
        hold_height=rng.uniform(0.03, max(0.03, min(0.08, body_h - 0.15))),
        seed=seed,
    )


class SimRobot(RobotInterface):
    """Speed-limited joint setpoints on the MuJoCo Stretch, plus a noisy 'camera'."""
    JOINT_ACT = {"lift": "lift", "arm": "arm", "wrist_yaw": "wrist_yaw",
                 "wrist_pitch": "wrist_pitch", "wrist_roll": "wrist_roll"}
    TOL = {"lift": 0.005, "arm": 0.005, "wrist_yaw": 0.03, "wrist_pitch": 0.03, "wrist_roll": 0.03}
    ACCEL = {"lift": 0.2, "arm": 0.2, "wrist_yaw": 2.0, "wrist_pitch": 2.0, "wrist_roll": 2.0}  # m/s^2, rad/s^2
    GRIPPER = {"open": 0.04, "closed": -0.02}

    def __init__(self, sim, rng, vision_noise):
        self.sim, self.m, self.d = sim, sim.m, sim.d
        self.rng, self.vision_noise = rng, vision_noise
        self.act = {k: self.m.actuator(v).id for k, v in self.JOINT_ACT.items()}
        self.sp = {}       # current (ramping) setpoint per joint
        self.vel = {}      # setpoint velocity (accel-limited profile)
        self.target = {}
        self.speed = {}
        self.bias = {k: 0.0 for k in self.act}  # integral term: removes steady-state sag
        self.grip_cmd = "closed"
        self.turn_target = None
        self.turn_speed = 0.5
        self.base = self.m.body("base_link").id
        self.slide = self.m.joint("joint_gripper_slide").qposadr[0]
        self.closed_empty = -0.011  # slide position when closed on nothing (measured)

    # --- state ---
    def joint(self, name):
        d, m = self.d, self.m
        if name == "arm":
            return sum(d.qpos[m.joint(f"joint_arm_l{i}").qposadr[0]] for i in range(4))
        jn = {"lift": "joint_lift", "wrist_yaw": "joint_wrist_yaw", "wrist_pitch": "joint_wrist_pitch",
              "wrist_roll": "joint_wrist_roll"}[name]
        return d.qpos[m.joint(jn).qposadr[0]]

    def base_yaw(self):
        R = self.d.xmat[self.base].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    dt = CTRL_DT

    def arm_effort(self):
        return float(self.d.actuator_force[self.act["arm"]])

    def gripper_holding(self):
        return self.grip_cmd == "closed" and self.d.qpos[self.slide] > self.closed_empty + 0.005

    hand_top_above_origin = 0.045  # m: top of the sim hand (palm/thumb) above the hand frame

    def estimate_bottle(self):
        """'Vision': true bottle axis / top and hand top in the base frame + Gaussian noise."""
        d, m = self.d, self.m
        R, o = d.xmat[self.base].reshape(3, 3), d.xpos[self.base]
        b = m.body("bottle").id
        axis = R.T @ (d.xpos[b] - o)
        top = d.xpos[b][2] + self.sim.sc.bottle.total_height
        hand_top = d.mocap_pos[m.body("person_hand").mocapid[0]][2] + self.hand_top_above_origin
        n, g = self.vision_noise, self.rng.gauss
        return {"axis_xy": (float(axis[0] + g(0, n)), float(axis[1] + g(0, n))),
                "top_z": float(top + g(0, n)), "hand_top_z": float(hand_top + g(0, n))}

    # --- commands ---
    def move(self, joint, target, speed):
        self.sp.setdefault(joint, self.joint(joint))
        self.vel.setdefault(joint, 0.0)
        self.target[joint], self.speed[joint] = target, speed

    def turn_base(self, angle, speed):
        self.turn_target, self.turn_speed = self.base_yaw() + angle, speed

    def set_gripper(self, opening):
        self.grip_cmd = opening
        self.d.ctrl[self.m.actuator("gripper").id] = self.GRIPPER[opening]

    def apply(self):
        """Called every control tick: ramp setpoints, run the base heading loop."""
        for j, tgt in self.target.items():
            # trapezoidal profile: accelerate, cruise at `speed`, decelerate into the target
            dist, a = tgt - self.sp[j], self.ACCEL[j]
            v_des = math.copysign(min(self.speed[j], math.sqrt(2 * a * abs(dist))), dist) if dist else 0.0
            dv = max(-a * CTRL_DT, min(a * CTRL_DT, v_des - self.vel[j]))
            self.vel[j] += dv
            new = self.sp[j] + self.vel[j] * CTRL_DT
            if (tgt - new) * dist <= 0:  # reached / passed the target
                new, self.vel[j] = tgt, 0.0
            self.sp[j] = new
            # integral correction for steady-state sag under load (the real robot's
            # controllers hold position; the sim's bare position servos don't)
            err = self.sp[j] - self.joint(j)
            self.bias[j] = max(-0.15, min(0.15, self.bias[j] + 8.0 * err * CTRL_DT))
            self.d.ctrl[self.act[j]] = self.sp[j] + self.bias.get(j, 0.0)
        w = 0.0
        if self.turn_target is not None:
            err = (self.turn_target - self.base_yaw() + math.pi) % (2 * math.pi) - math.pi
            w = max(-self.turn_speed, min(self.turn_speed, 2.5 * err))
            if abs(err) < 0.004:
                w = 0.0
        from stretch_mujoco.utils import diff_drive_inv_kinematics
        wl, wr = diff_drive_inv_kinematics(0.0, w)
        self.d.ctrl[self.m.actuator("left_wheel_vel").id] = wl
        self.d.ctrl[self.m.actuator("right_wheel_vel").id] = wr

    def motion_done(self):
        for j, tgt in self.target.items():
            if abs(self.sp[j] - tgt) > 1e-9 or abs(self.joint(j) - tgt) > self.TOL[j]:
                return False
        if self.turn_target is not None:
            err = (self.turn_target - self.base_yaw() + math.pi) % (2 * math.pi) - math.pi
            ang_vel = abs(self.d.qvel[self.m.jnt_dofadr[self.m.body_jntadr[self.base]] + 5])
            if abs(err) > 0.01 or ang_vel > 0.02:
                return False
        return True

    def wait_motion(self, timeout=15.0):
        t0 = self.d.time
        settle = 0
        while self.d.time - t0 < timeout:
            settle = settle + 1 if self.motion_done() else 0
            if settle >= 10:
                return True
            yield
        return False

    def wait_gripper(self, timeout=3.0):
        t0, still = self.d.time, 0
        dof = self.m.joint("joint_gripper_slide").dofadr[0]
        while self.d.time - t0 < timeout:
            still = still + 1 if (self.d.time - t0 > 0.15 and abs(self.d.qvel[dof]) < 0.003) else 0
            if still >= 10:
                return True
            yield
        return False

    def wait(self, seconds):
        t0 = self.d.time
        while self.d.time - t0 < seconds:
            yield


class Person:
    """Holds the bottle out (with a little tremor), lets go after the robot grips, takes it back when offered."""

    def __init__(self, sim, sc: Scenario, rng):
        self.sim, self.sc, self.rng = sim, sc, rng
        m = sim.m
        self.hand = m.body("person_hand").mocapid[0]
        self.torso = m.body("person_torso").mocapid[0]
        self.state = "HOLDING"
        self.t_event = None
        self.phase_f = [rng.uniform(0, 6.28) for _ in range(3)]
        self.freq = [rng.uniform(0.3, 0.8) for _ in range(3)]
        az = math.radians(sc.bottle_azimuth)
        self.dir = np.array([math.cos(az), math.sin(az), 0.0])   # robot -> bottle
        self.hand_yaw = math.atan2(-self.dir[1], -self.dir[0])  # hand +x points at the robot
        self.grip_offset_z = sc.hold_height                       # hand grip height above bottle bottom
        bottom_z = sc.grasp_height - (sc.hold_height + 0.045 + 0.07)  # puts the robot's grasp near grasp_height
        self.hold_pos = np.array([sc.bottle_rho * self.dir[0], sc.bottle_rho * self.dir[1], bottom_z + self.grip_offset_z])
        self.away_pos = self.hold_pos + 0.10 * self.dir + np.array([0, 0, -0.30])  # lower the arm, toward the person
        self.pos = self.hold_pos.copy()
        self.move_from, self.move_to, self.move_t0, self.move_dur = None, None, 0, 1

    def place(self):
        d, m = self.sim.d, self.sim.m
        d.mocap_pos[self.hand] = self.hold_pos
        d.mocap_quat[self.hand] = yaw_quat(self.hand_yaw)
        d.mocap_pos[self.torso] = self.hold_pos * [1, 1, 0] + 0.42 * self.dir
        d.mocap_quat[self.torso] = yaw_quat(self.hand_yaw)
        qa = m.jnt_qposadr[m.joint("bottle_free").id]
        d.qpos[qa:qa + 3] = self.hold_pos - [0, 0, self.grip_offset_z]
        d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        set_weld(m, d, True)

    def _go(self, target, dur):
        self.move_from, self.move_to, self.move_t0, self.move_dur = self.pos.copy(), np.array(target), self.sim.d.time, dur

    def update(self, ctrl_phase, robot: SimRobot):
        d, m, t = self.sim.d, self.sim.m, self.sim.d.time
        if self.state == "HOLDING" and ctrl_phase == "TAKE":
            self.state, self.t_event = "ABOUT_TO_RELEASE", t
        elif self.state == "ABOUT_TO_RELEASE" and t - self.t_event >= self.sc.release_delay:
            set_weld(m, d, False)
            self.state = "WITHDRAWING"
            self._go(self.away_pos, 1.2)
        elif self.state == "WITHDRAWING" and ctrl_phase == "GIVE" and robot.motion_done():
            self.state, self.t_event = "WAITING_TO_GRAB", t
        elif self.state == "WAITING_TO_GRAB" and t - self.t_event >= self.sc.grab_delay:
            b = d.xpos[m.body("bottle").id]
            self.state = "REACHING"
            self._go(b + [0, 0, self.grip_offset_z], 1.0)
        elif self.state == "REACHING" and t - self.move_t0 >= self.move_dur:
            set_weld(m, d, True)
            self.state, self.t_event = "HAS_BOTTLE", t
        elif self.state == "HAS_BOTTLE" and t - self.t_event >= 0.4:
            self.state = "TUGGING"           # keep pulling gently (up to 4 cm) until the robot lets go
            self._go(self.pos + 0.04 * self.dir, 1.5)
        elif self.state == "TUGGING" and ctrl_phase in ("STOW", "DONE"):
            self.state = "LEAVING"
            self._go(self.pos + 0.08 * self.dir + [0, 0, -0.25], 1.5)

        if self.move_to is not None:
            s = min(1.0, (t - self.move_t0) / self.move_dur)
            s = s * s * (3 - 2 * s)
            self.pos = self.move_from + s * (self.move_to - self.move_from)
        tremor = np.array([self.sc.tremor * math.sin(2 * math.pi * f * t + p) for f, p in zip(self.freq, self.phase_f)])
        if self.state in ("HOLDING", "ABOUT_TO_RELEASE", "HAS_BOTTLE", "TUGGING"):
            d.mocap_pos[self.hand] = self.pos + tremor
        else:
            d.mocap_pos[self.hand] = self.pos


class Checks:
    """Safety and task checks, evaluated every control tick."""

    def __init__(self, sim):
        m = sim.m
        self.sim = sim
        base = m.body("base_link").id
        self.robot = m.body_rootid[m.geom_bodyid] == base
        hand = m.body("person_hand").id
        self.person = m.geom_bodyid == hand
        self.bottle_g = m.geom_bodyid == m.body("bottle").id
        names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self.finger = np.array([("finger" in n or "rubber_tip" in n) for n in names])
        # gripper = fingers + the gripper's palm: pressing the bottle against the palm is part
        # of grasping. Anything else (wrist, arm, mast) touching the bottle is a failure.
        self.gripper = np.array([("finger" in n or "rubber_tip" in n or "gripper" in n) for n in names])
        self.self_pairs0 = None      # robot-robot contact pairs present at the start (resting contacts)
        self.self_new = set()
        self.person_contact_steps = 0
        self.person_contact_parts = set()
        self.nonfinger_bottle_steps = 0
        self.nonfinger_parts = set()
        self.max_tilt_held = 0.0
        self.max_slip = 0.0
        self.dropped = False
        self.ref_rel = None
        self.gc = m.body("link_grasp_center").id
        self.bsite = m.site("bottle_grasp_ref").id
        self.bbody = m.body("bottle").id

    def update(self, phase, robot_holding):
        m, d = self.sim.m, self.sim.d
        touching_person = False
        pairs = set()
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if self.robot[g1] and self.robot[g2] and not (self.finger[g1] and self.finger[g2]):
                # (finger-on-finger is just the gripper closed on nothing)
                pairs.add(tuple(sorted((m.body(m.geom_bodyid[g1]).name, m.body(m.geom_bodyid[g2]).name))))
            if (self.robot[g1] and self.person[g2]) or (self.robot[g2] and self.person[g1]):
                touching_person = True
                rg = g1 if self.robot[g1] else g2
                self.person_contact_parts.add(m.body(m.geom_bodyid[rg]).name)
            for a, b in ((g1, g2), (g2, g1)):
                if self.robot[a] and self.bottle_g[b] and not self.gripper[a]:
                    self.nonfinger_bottle_steps += 1
                    self.nonfinger_parts.add(m.body(m.geom_bodyid[a]).name)
        if self.self_pairs0 is None:
            self.self_pairs0 = pairs
        self.self_new |= pairs - self.self_pairs0
        self.person_contact_steps += touching_person
        bz = d.xmat[self.bbody].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, bz[2]))))
        if d.xpos[self.bbody][2] < 0.15:
            self.dropped = True
        if phase in ("RETRACT", "HOLD", "GIVE"):
            self.max_tilt_held = max(self.max_tilt_held, tilt)
            Rg = d.xmat[self.gc].reshape(3, 3)
            rel = Rg.T @ (d.site_xpos[self.bsite] - d.xpos[self.gc])
            if self.ref_rel is None:
                self.ref_rel = rel
            self.max_slip = max(self.max_slip, float(np.linalg.norm(rel - self.ref_rel)))


class Sim:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.m = build_model(sc.bottle, sc.grip_force)
        self.d = mujoco.MjData(self.m)
        A = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        # start stowed, like the real robot: lift 0.3, arm in, wrist tucked
        for k, v in {"lift": 0.3, "arm": 0.0, "wrist_yaw": math.pi, "wrist_pitch": -0.4, "wrist_roll": 0.0,
                     "gripper": 0.0, "head_pan": -1.6, "head_tilt": -0.5}.items():
            self.d.ctrl[A[k]] = v
        for _ in range(int(1.5 / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        self.d.time = 0.0


def run_trial(sc: Scenario, video=None, viewer=False, give_back=True, verbose=False):
    rng = random.Random(sc.seed)
    sim = Sim(sc)
    m, d = sim.m, sim.d
    robot = SimRobot(sim, rng, sc.vision_noise)
    person = Person(sim, sc, rng)
    person.place()
    checks = Checks(sim)
    res = Result()
    gen = run_handoff(robot, Kinematics(), Limits(), res, give_back=give_back)
    steps = max(1, int(round(CTRL_DT / m.opt.timestep)))

    renderer, writer, cam, next_frame = None, None, None, 0.0
    if video:
        import cv2
        renderer = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        az = math.radians(sc.bottle_azimuth)
        mid = np.array([0.5 * sc.bottle_rho * math.cos(az), 0.5 * sc.bottle_rho * math.sin(az), 0.75])
        cam.lookat[:] = mid
        cam.distance, cam.elevation = 2.0, -16
        cam.azimuth = math.degrees(az) + 70
        vis = mujoco.MjvOption()
        vis.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0  # hide the lidar rays
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    view = None
    if viewer:
        from mujoco import viewer as mj_viewer
        view = mj_viewer.launch_passive(m, d)
        wall0 = time.time()

    done = False
    t_end_extra = None
    while True:
        if not done:
            try:
                next(gen)
            except StopIteration:
                done = True
                t_end_extra = d.time + 2.0
        if done and d.time >= t_end_extra:
            break
        if d.time > 150:
            res.reason = res.reason or "timeout"
            break
        robot.apply()
        person.update(res.phase, robot)
        for _ in range(steps):
            mujoco.mj_step(m, d)
        checks.update(res.phase, robot.gripper_holding())
        if writer is not None and d.time >= next_frame:
            import cv2
            renderer.update_scene(d, cam, scene_option=vis)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            label = {"FACE": "turning to face the bottle", "RAISE": "raising to grasp height",
                     "UNFOLD": "opening gripper", "REACH": "reaching", "GRASP": "grasping",
                     "TAKE": "taking the bottle", "RETRACT": "holding the bottle upright", "HOLD": "holding",
                     "GIVE": "handing it back", "RELEASE": "releasing", "STOW": "stowing", "DONE": "done"}.get(res.phase, res.phase.lower())
            cv2.putText(img, "Stretch bottle handoff (MuJoCo simulation)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(img, f"{label}   t={d.time:5.1f}s", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 90, 200), 2, cv2.LINE_AA)
            writer.write(img)
            next_frame += 1 / 30
        if view is not None:
            if not view.is_running():
                break
            view.sync()
            lag = d.time - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)

    if writer is not None:
        writer.release()
    if view is not None:
        view.close()

    bottle_z = float(d.xpos[m.body("bottle").id][2])
    hb_ok = (not give_back) or (person.state in ("TUGGING", "LEAVING") and d.eq_active[m.equality("hand_holds_bottle").id] == 1)
    checks_ok = {
        "behavior_finished": res.ok,
        "no_motion_timeouts": res.timeouts == 0,
        "no_drop": not checks.dropped,
        "never_touched_person": checks.person_contact_steps == 0,
        "no_self_collision": not checks.self_new,
        "only_gripper_touched_bottle": checks.nonfinger_bottle_steps == 0,
        "bottle_upright_while_held": checks.max_tilt_held < 10.0,
        "no_slip_while_held": checks.max_slip < 0.02,
        "handed_back": hb_ok,
    }
    checks_ok = {k: bool(v) for k, v in checks_ok.items()}
    out = {
        "passed": all(checks_ok.values()),
        "checks": checks_ok,
        "reason": res.reason,
        "phase_reached": res.phase,
        "sim_time_s": round(d.time, 1),
        "max_tilt_deg": round(checks.max_tilt_held, 1),
        "max_slip_cm": round(100 * checks.max_slip, 2),
        "person_contact_parts": sorted(checks.person_contact_parts),
        "self_collisions": sorted(checks.self_new),
        "nonfinger_bottle_parts": sorted(checks.nonfinger_parts),
        "bottle_final_z": round(bottle_z, 3),
        "plan": None if res.plan is None else {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(res.plan).items()},
        "scenario": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(sc).items() if k != "bottle"},
        "bottle": {k: round(v, 4) for k, v in asdict(sc.bottle).items()},
    }
    if verbose:
        print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=0, help="randomized batch of N trials")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", help="write an mp4 of the default (or --seed) scenario")
    ap.add_argument("--viewer", action="store_true", help="watch live in the MuJoCo viewer")
    ap.add_argument("--random", action="store_true", help="use a randomized scenario for --video/--viewer")
    ap.add_argument("--azimuth", type=float, default=-45.0,
                    help="direction of the person from the robot for --video/--viewer (deg; -90 = robot's right)")
    ap.add_argument("--out", default=os.path.expanduser("~/ros2_ws/artifacts/handoff_batch.json"))
    args = ap.parse_args()

    if args.trials:
        rng = random.Random(args.seed)
        results = []
        for i in range(args.trials):
            sc = random_scenario(rng, seed=args.seed * 1000 + i)
            r = run_trial(sc)
            results.append(r)
            bad = [k for k, v in r["checks"].items() if not v]
            print(f"trial {i:3d}: {'PASS' if r['passed'] else 'FAIL'}  {r['sim_time_s']:5.1f}s  "
                  f"tilt {r['max_tilt_deg']:4.1f}  slip {r['max_slip_cm']:4.2f}cm  {bad} {r['reason']}", flush=True)
        n = sum(r["passed"] for r in results)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n{n}/{len(results)} passed  (details: {args.out})")
        sys.exit(0 if n == len(results) else 1)

    sc = random_scenario(random.Random(args.seed), args.seed) if args.random else Scenario(seed=args.seed, bottle_azimuth=args.azimuth)
    r = run_trial(sc, video=args.video, viewer=args.viewer, verbose=True)
    if args.video:
        print("video:", args.video)
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
