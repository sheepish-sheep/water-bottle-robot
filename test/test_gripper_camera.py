"""Gripper camera: bottle position from its box (bottle_geometry) and the detector node's
per-frame processing, with a fake detector (no YOLO, no ROS).

    python3 -m pytest test/test_gripper_camera.py -q
"""
import math

import numpy as np
import pytest

from water_bottle_robot.perception.bottle_geometry import Intrinsics, box_to_point, pick_bottle
from water_bottle_robot.perception import gripper_bottle_detector as gbd

# roughly the D405 color stream on Stretch (424x240)
K = Intrinsics(fx=215.0, fy=215.0, cx=212.0, cy=120.0, width=424, height=240)


def silhouette_box(x, y, z, r=0.035, half_h=0.04):
    """Exact image box of a vertical cylinder (axis along the camera's y) at (x, y, z):
    the left/right edges are the tangent lines from the camera."""
    rho, ang = math.hypot(x, z), math.atan2(x, z)
    half = math.asin(r / rho)
    xl, xr = math.tan(ang - half), math.tan(ang + half)
    return (K.cx + K.fx * xl, K.cy + K.fy * (y - half_h) / z, K.cx + K.fx * xr, K.cy + K.fy * (y + half_h) / z, 0.9)


@pytest.mark.parametrize("x, z", [(0.0, 0.20), (0.04, 0.25), (-0.06, 0.30), (0.02, 0.45)])
def test_position_from_the_box(x, z):
    p = box_to_point(silhouette_box(x, 0.01, z), K, 0.07)
    assert p[2] == pytest.approx(z, rel=0.02)                 # distance within 2%
    assert p[0] == pytest.approx(x, abs=0.003)                # sideways within 3 mm


def test_wrong_diameter_scales_the_distance():
    p = box_to_point(silhouette_box(0.0, 0.0, 0.30), K, 0.077)   # bottle is 7 cm, told 7.7
    assert p[2] == pytest.approx(0.33, rel=0.02)


def test_box_cut_off_at_the_image_edge_gives_no_distance():
    assert box_to_point((0, 50, 120, 200, 0.9), K, 0.07) is None
    assert box_to_point((300, 50, K.width - 1, 200, 0.9), K, 0.07) is None


def test_tracks_the_same_bottle():
    a, b = (10, 10, 60, 100, 0.9), (200, 10, 300, 100, 0.8)   # b is wider (nearer)
    assert pick_bottle([a, b]) == b
    assert pick_bottle([a, b], previous=(12, 12, 62, 98, 0.9)) == a
    assert pick_bottle([]) is None


def test_process_and_annotate():
    img = np.zeros((K.height, K.width, 3), np.uint8)
    box = silhouette_box(0.0, 0.0, 0.25)
    got_box, point = gbd.process(img, K, lambda _: [box], 0.07)
    assert got_box == box and point[2] == pytest.approx(0.25, rel=0.02)
    assert gbd.annotate(img, got_box, point).shape == img.shape
    assert gbd.process(img, K, lambda _: [], 0.07) == (None, None)
    assert gbd.annotate(img, None, None).shape == img.shape


def test_z_axis_of_quaternion():
    from water_bottle_robot.perception.bottle_geometry import z_axis_of
    assert z_axis_of(0, 0, 0, 1) == pytest.approx((0, 0, 1))
    s = math.sqrt(0.5)                                   # 90 deg about x: z -> -y
    assert z_axis_of(s, 0, 0, s) == pytest.approx((0, -1, 0))


def test_gripper_camera_geometry_in_mujoco():
    """Stretch's D405 as mounted in the robot model, posed the way ALIGN looks (arm 15 cm
    short of the grasp, wrist straight): render it, box the bottle's pixels, and check
    the whole chain (box -> camera point -> world) against where the bottle really is.
    Catches a rotated image, wrong frame conventions, and the camera-tilt bias."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = pytest.importorskip("mujoco")
    from water_bottle_robot.manipulation.handoff_scene import build_model, BottleSpec
    from water_bottle_robot.manipulation.handoff_controller import Kinematics, Limits, plan_reach
    try:
        kin, lim, spec = Kinematics(), Limits(), BottleSpec()
        m = build_model(spec)
        d = mujoco.MjData(m)
        W, H = 424, 240
        r = mujoco.Renderer(m, H, W)
    except Exception as e:      # no MuJoCo model / no EGL on this machine
        pytest.skip(f"MuJoCo rendering unavailable: {e}")
    cam = m.camera("d405_rgb").id
    f = (H / 2) / math.tan(math.radians(m.cam_fovy[cam]) / 2)
    Kc = Intrinsics(fx=f, fy=f, cx=W / 2, cy=H / 2, width=W, height=H)

    def setj(name, v):
        d.qpos[m.joint(name).qposadr[0]] = v
    for off_x, dist in [(0.0, 0.70), (0.04, 0.70), (-0.04, 0.66), (0.02, 0.76)]:
        p = plan_reach((kin.gx, -dist, 0.90), kin, lim)
        mujoco.mj_resetData(m, d)
        setj("joint_lift", p.lift)
        for i in range(4):
            setj(f"joint_arm_l{i}", (p.arm_grasp - lim.align_look_back) / 4)
        for j in ("joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"):
            setj(j, 0.0)
        a = m.joint("bottle_free").qposadr[0]
        d.qpos[a:a + 7] = (kin.gx + off_x, -dist, 0.90 - spec.body_height * 0.75, 1, 0, 0, 0)
        mujoco.mj_forward(m, d)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam)
        seg = r.render()
        r.disable_segmentation_rendering()
        mask = (seg[..., 0] == m.geom("bottle_body").id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        ys, xs = np.nonzero(mask)
        assert len(xs), "bottle not in the gripper camera's view"
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 1.0)
        assert box[3] - box[1] > box[2] - box[0]          # upright in the image (not rotated 90 deg)
        R = d.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1])   # MuJoCo camera -> ROS optical frame
        up = R.T @ np.array([0, 0, 1.0])
        world = d.cam_xpos[cam] + R @ np.array(box_to_point(box, Kc, 2 * spec.radius, up))
        assert world[0] == pytest.approx(kin.gx + off_x, abs=0.004)   # off the arm's line: what the base fixes
        assert world[1] == pytest.approx(-dist, abs=0.004)            # along the arm: what the reach fixes
    r.close()
