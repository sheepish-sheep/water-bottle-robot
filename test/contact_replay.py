#!/usr/bin/env python3
"""
Real collision check for a Nav2 run: replay a recorded path through the same
MuJoCo kitchen (robot in the settled 'stow' pose) and measure how close any
robot part gets to the scene. Catches things the lidar can't see, like the
base's back corner or the wrist hitting a countertop edge.

    python3 test/contact_replay.py RUN_trace.json --spawn X Y YAW --obstacles "x,y" [--min-gap 0.02]

Exit 0 if every robot part stayed at least --min-gap (m) from the scene.
"""
import argparse
import json
import math
import os
import random
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from spawn_driver import add_obstacles  # noqa: E402

MARGIN = 0.10  # m: report anything within 10 cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--spawn", nargs=3, type=float, required=True, metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--obstacles", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gap", type=float, default=0.02)
    args = ap.parse_args()

    from stretch_mujoco import robocasa_gen
    random.seed(args.seed); np.random.seed(args.seed)  # same scene as spawn_driver.py
    orig = robocasa_gen.add_stretch_to_kitchen
    robocasa_gen.add_stretch_to_kitchen = lambda xml, pose: orig(add_obstacles(xml, args.obstacles)[0], pose)
    sx, sy, syaw = args.spawn
    h = math.radians(syaw) / 2
    m, _, _ = robocasa_gen.model_generation_wizard(
        task="PnPCounterToCab", layout=7, style=3,
        robot_spawn_pose={"pos": f"{sx} {sy} 0", "quat": f"{math.cos(h)} 0 0 {math.sin(h)}"})
    d = mujoco.MjData(m)
    base = m.body("base_link").id
    qa = m.jnt_qposadr[m.body_jntadr[base]]
    d.ctrl[:] = m.key("stow").ctrl          # the keyframe only sets actuator targets
    for _ in range(int(4.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z0 = d.qpos[qa + 2]
    is_robot = m.body_rootid[m.geom_bodyid] == base
    m.geom_margin[:] = np.maximum(m.geom_margin, MARGIN)

    odom = json.load(open(args.trace))["odom"]
    t0 = odom[0][0] if odom else 0.0
    worst = {}  # (robot body, scene body) -> (gap, t, pose)
    for t, x, y, yaw in odom[::2]:
        d.qpos[qa:qa + 3] = [x, y, z0]
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            r1, r2 = is_robot[c.geom1], is_robot[c.geom2]
            if r1 == r2:
                continue
            rg, sg = (c.geom1, c.geom2) if r1 else (c.geom2, c.geom1)
            if m.geom_type[sg] == mujoco.mjtGeom.mjGEOM_PLANE or "floor" in m.body(m.geom_bodyid[sg]).name:
                continue  # the floor (RoboCasa's is a box, not a plane)
            if m.geom_bodyid[rg] != base and "wheel" in m.body(m.geom_bodyid[rg]).name:
                continue
            key = (m.body(m.geom_bodyid[rg]).name, m.body(m.geom_bodyid[sg]).name)
            if key not in worst or c.dist < worst[key][0]:
                worst[key] = (c.dist, round(t - t0, 1), (round(x, 2), round(y, 2), round(math.degrees(yaw))))

    gap = min((v[0] for v in worst.values()), default=MARGIN)
    for (rb, sb), (dist, t, pose) in sorted(worst.items(), key=lambda kv: kv[1][0])[:8]:
        print(f"  {dist * 100:6.1f} cm  t={t:5.1f}s pose={pose}  robot:{rb}  scene:{sb}")
    ok = gap >= args.min_gap
    print(json.dumps({"closest_robot_to_scene_m": round(gap, 3), "min_gap_m": args.min_gap,
                      "samples": len(odom[::2]), "passed": ok}))
    print("CONTACT CHECK PASS" if ok else "CONTACT CHECK FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
