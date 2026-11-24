"""ROS plumbing for the gripper camera, without a robot, sim or YOLO:

- gripper_bottle_detector: camera image + camera_info in, /gripper_bottle/point out
  (QoS, cv_bridge encodings, the TF lookup for the tilt correction).
- handoff_ros: a /gripper_bottle/point in the camera frame -> TF -> base_link estimate.

    python3 -m pytest test/test_gripper_camera_ros.py -q     # needs ROS + ~/ament_ws sourced
"""
import math
import time
import types

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
tf2_ros = pytest.importorskip("tf2_ros")
from geometry_msgs.msg import PointStamped, TransformStamped  # noqa: E402

CAM = "gripper_camera_color_optical_frame"
# Camera looking straight along base +x, level: optical z = base +x, x = base -y, y = base -z.
OPTICAL = (-0.5, 0.5, -0.5, 0.5)   # quaternion (x, y, z, w) of base_link -> optical frame


def camera_tf(stamp, xyz=(0.1, -0.5, 1.0)):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = "base_link", CAM
    t.header.stamp = stamp
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    q = t.transform.rotation
    q.x, q.y, q.z, q.w = OPTICAL
    return t


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_detector_node_publishes_the_bottle(ros):
    from cv_bridge import CvBridge
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import CameraInfo, Image
    from water_bottle_robot.perception import gripper_bottle_detector as gbd

    W, H, f = 424, 240, 215.0
    # a 7 cm bottle 25 cm straight ahead: its silhouette is 2*atan(r/rho) wide
    half = math.tan(math.asin(0.035 / 0.25))
    box = (W / 2 - f * half, 60.0, W / 2 + f * half, 180.0, 0.9)
    node = gbd.create_node(detect=lambda img: [box])
    helper = rclpy.create_node("test_helper")
    got = []
    helper.create_subscription(PointStamped, "/gripper_bottle/point", got.append, 10)
    pub_img = helper.create_publisher(Image, "/gripper_camera/color/image_rect_raw", 10)
    pub_info = helper.create_publisher(CameraInfo, "/gripper_camera/color/camera_info", 10)
    tf_pub = tf2_ros.StaticTransformBroadcaster(helper)
    tf_pub.sendTransform(camera_tf(helper.get_clock().now().to_msg()))

    info = CameraInfo(width=W, height=H)
    info.header.frame_id = CAM
    info.k = [f, 0.0, W / 2, 0.0, f, H / 2, 0.0, 0.0, 1.0]
    img = CvBridge().cv2_to_imgmsg(np.zeros((H, W, 3), np.uint8), "rgb8")   # sim publishes rgb8
    img.header.frame_id = CAM

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(helper)
    t0 = time.time()
    while not got and time.time() - t0 < 10.0:
        pub_info.publish(info)
        pub_img.publish(img)
        ex.spin_once(timeout_sec=0.05)
    node.destroy_node()
    helper.destroy_node()
    assert got, "no /gripper_bottle/point"
    p = got[-1]
    assert p.header.frame_id == CAM
    assert p.point.z == pytest.approx(0.25, rel=0.02) and p.point.x == pytest.approx(0.0, abs=0.003)


def test_handoff_ros_puts_camera_points_in_base_link(ros, monkeypatch):
    pytest.importorskip("hello_helpers.gripper_conversion")
    from water_bottle_robot.manipulation import handoff_real, handoff_ros
    from water_bottle_robot.manipulation.handoff_controller import Kinematics, Limits

    buf = tf2_ros.Buffer()
    buf.set_transform_static(camera_tf(rclpy.time.Time().to_msg()), "test")
    warnings = []
    node = types.SimpleNamespace(tf2_buffer=buf,
                                 get_logger=lambda: types.SimpleNamespace(warn=lambda m, **k: warnings.append(m)))
    kin, lim = Kinematics(), Limits()
    rr = handoff_ros.RosRobot(node, handoff_real.fixed_spot(0.70, 0.83, kin, lim))
    assert rr.see_bottle() is None

    msg = PointStamped()
    msg.header.frame_id = CAM
    msg.point.x, msg.point.y, msg.point.z = 0.02, 0.0, 0.30   # 30 cm ahead, 2 cm to the image's right
    rr.on_bottle_point(msg)
    seq, (x, y, z) = rr.see_bottle()
    assert (x, y, z) == pytest.approx((0.1 + 0.30, -0.5 - 0.02, 1.0))   # image right = base -y

    rr.on_bottle_point(msg)
    assert rr.see_bottle()[0] == seq + 1                         # a new estimate each message

    now = time.time()
    monkeypatch.setattr(handoff_ros.time, "time", lambda: now + 1.0)
    assert rr.see_bottle() is None                               # stale estimates don't count

    msg.header.frame_id = "no_such_frame"
    rr.on_bottle_point(msg)                                      # no TF: skipped with a warning
    assert warnings
