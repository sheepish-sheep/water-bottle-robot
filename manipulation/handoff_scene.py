"""
MuJoCo scene for the bottle-handoff demo: Stretch, a clear water bottle, and a
simple "person" whose hand holds the bottle out to the robot.

The person's hand is a mocap body (moved by the script, like a real hand the
robot can't push around). While the person is holding the bottle, a weld
constraint ties the bottle to the hand; releasing = turning the weld off.
The hand and forearm collide with the robot, so any robot-person contact shows
up in the safety check. The torso and head are visual only.
"""
import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class BottleSpec:
    """A clear plastic bottle. Defaults: ~750 ml reusable bottle."""
    radius: float = 0.036        # m (7.2 cm diameter)
    body_height: float = 0.20    # m, straight cylindrical part
    mass: float = 0.08           # kg (empty ~0.05-0.1; full 750 ml ~0.8)
    friction: float = 0.6        # sliding friction vs the rubber fingertips
    tape_height: float = 0.17    # m above the base: the fill-line tape (opaque)

    @property
    def total_height(self):
        return self.body_height + 0.03 + 0.02 + 0.02  # shoulder + neck + cap


def _bottle_xml(b: BottleSpec):
    r, h = b.radius, b.body_height
    clear = 'rgba="0.80 0.92 1.0 0.30"'
    # collision: main body + shoulder + neck + cap. Mass is put on the main body.
    return f"""
    <body name="bottle" pos="0 3 0.3">
      <freejoint name="bottle_free"/>
      <geom name="bottle_body" type="cylinder" size="{r} {h/2}" pos="0 0 {h/2}" mass="{b.mass}"
            friction="{b.friction} 0.02 0.001" condim="4" {clear} group="1"/>
      <geom name="bottle_shoulder" type="cylinder" size="{r*0.75} 0.015" pos="0 0 {h + 0.015}" mass="0.001"
            friction="{b.friction} 0.02 0.001" condim="4" {clear} group="1"/>
      <geom name="bottle_neck" type="cylinder" size="{r*0.45} 0.01" pos="0 0 {h + 0.04}" mass="0.001"
            friction="{b.friction} 0.02 0.001" {clear} group="1"/>
      <geom name="bottle_cap" type="cylinder" size="{r*0.5} 0.01" pos="0 0 {h + 0.06}" mass="0.005"
            rgba="0.15 0.35 0.85 1" group="1"/>
      <geom name="bottle_tape" type="cylinder" size="{r + 0.0008} 0.006" pos="0 0 {b.tape_height}"
            contype="0" conaffinity="0" mass="0" rgba="0.95 0.95 0.95 1" group="1"/>
      <geom name="bottle_water_hint" type="cylinder" size="{r*0.97} {h*0.15}" pos="0 0 {h*0.15}"
            contype="0" conaffinity="0" mass="0" rgba="0.35 0.6 1.0 0.25" group="1"/>
      <site name="bottle_grasp_ref" pos="0 0 {h*0.75}" size="0.004"/>
    </body>"""


def _person_xml():
    skin = 'rgba="0.86 0.68 0.56 1"'
    shirt = 'rgba="0.30 0.45 0.35 1"'
    # hand frame: origin at the bottle axis at the hand's grip height; local +x
    # points from the person toward the robot, so the palm sits at -x (behind
    # the bottle) and the forearm runs back toward the torso along -x.
    return f"""
    <body name="person_hand" mocap="true" pos="0 3 1">
      <geom name="hand_palm" type="box" size="0.012 0.045 0.045" pos="-0.050 0 0" {skin}/>
      <geom name="hand_fingers_l" type="capsule" fromto="-0.04 0.035 0.00  0.025 0.042 0.00" size="0.011" {skin}/>
      <geom name="hand_fingers_r" type="capsule" fromto="-0.04 -0.035 0.00  0.025 -0.042 0.00" size="0.011" {skin}/>
      <geom name="hand_thumb" type="capsule" fromto="-0.04 0.035 0.03  0.0 0.045 0.035" size="0.010" {skin}/>
      <geom name="forearm" type="capsule" fromto="-0.07 0 -0.01  -0.33 0 -0.12" size="0.030" {skin}/>
    </body>
    <body name="person_torso" mocap="true" pos="0 3 0">
      <geom name="torso" type="capsule" fromto="0 0 0.95  0 0 1.40" size="0.17" {shirt} contype="0" conaffinity="0"/>
      <geom name="hips" type="capsule" fromto="0 -0.1 0.92  0 0.1 0.92" size="0.15" {shirt} contype="0" conaffinity="0"/>
      <geom name="leg_l" type="capsule" fromto="0 0.09 0.85  0 0.09 0.05" size="0.07" rgba="0.2 0.22 0.3 1" contype="0" conaffinity="0"/>
      <geom name="leg_r" type="capsule" fromto="0 -0.09 0.85  0 -0.09 0.05" size="0.07" rgba="0.2 0.22 0.3 1" contype="0" conaffinity="0"/>
      <geom name="head" type="sphere" size="0.11" pos="0 0 1.62" {skin} contype="0" conaffinity="0"/>
      <geom name="upper_arm" type="capsule" fromto="0 -0.2 1.35  0.12 -0.24 1.08" size="0.045" {shirt} contype="0" conaffinity="0"/>
    </body>"""


def build_model(bottle: BottleSpec = BottleSpec(), grip_force: float = 30.0):
    """
    Returns a compiled MjModel. grip_force caps the gripper slide actuator force
    (N), standing in for the real gripper's limited squeeze; trials vary it to
    make sure the grasp doesn't depend on an unrealistically strong grip.
    """
    from stretch_mujoco.utils import get_absolute_path_stretch_xml
    robot = get_absolute_path_stretch_xml({"pos": "0 0 0", "quat": "1 0 0 0"})
    xml = f"""<mujoco model="stretch bottle handoff">
  <include file="{robot}"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <texture type="skybox" builtin="gradient" rgb1="0.9 0.93 0.97" rgb2="0.6 0.68 0.78" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0.5 -1 2.5" dir="-0.2 0.4 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <light pos="-1 1 2" dir="0.4 -0.4 -1" directional="true" castshadow="false" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"/>
    {_bottle_xml(bottle)}
    {_person_xml()}
  </worldbody>
  <contact>
    <exclude body1="person_hand" body2="bottle"/>
  </contact>
  <equality>
    <weld name="hand_holds_bottle" body1="person_hand" body2="bottle" solref="0.005 1" active="false"/>
  </equality>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    fix_wheels(m)
    g = m.actuator("gripper")
    m.actuator_forcelimited[g.id] = 1
    m.actuator_forcerange[g.id] = [-grip_force, grip_force]
    return m


WHEEL_KV = 180.0


def fix_wheels(m):
    """Same fix as sim/spawn_driver.py: stock gear=3 makes the base ~6x too slow."""
    for name in ("left_wheel_vel", "right_wheel_vel"):
        a = m.actuator(name)
        a.gear[0] = 1.0
        a.gainprm[0] = WHEEL_KV
        a.biasprm[2] = -WHEEL_KV


def set_weld(m, d, active, name="hand_holds_bottle"):
    """Turn the hand-bottle weld on/off. Turning it on holds the CURRENT relative pose."""
    eq = m.equality(name).id
    if active:
        h, b = m.body("person_hand").id, m.body("bottle").id
        mujoco.mj_kinematics(m, d)
        # relpose = pose of body2 (bottle) in body1 (hand) frame
        Rh = d.xmat[h].reshape(3, 3)
        rel_pos = Rh.T @ (d.xpos[b] - d.xpos[h])
        qh_inv = np.zeros(4); mujoco.mju_negQuat(qh_inv, d.xquat[h])
        rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, qh_inv, d.xquat[b])
        m.eq_data[eq][0:3] = 0.0            # anchor (unused for weld rel pose)
        m.eq_data[eq][3:6] = rel_pos
        m.eq_data[eq][6:10] = rel_q
        m.eq_data[eq][10] = 1.0             # torquescale
    d.eq_active[eq] = 1 if active else 0


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
