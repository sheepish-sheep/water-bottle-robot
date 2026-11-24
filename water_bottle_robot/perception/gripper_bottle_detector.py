"""
ROS 2 node: find the bottle in Stretch's gripper camera (D405) with YOLO and
publish where it is.

    # robot: the gripper camera, then this node
    ros2 launch stretch_core d405_basic.launch.py
    ros2 run water_bottle_robot gripper_bottle_detector
    # watch what it sees (any machine on the same ROS network)
    ros2 run rqt_image_view rqt_image_view /gripper_bottle/image

    # MuJoCo sim (stretch_simulation names the topics differently)
    ros2 run water_bottle_robot gripper_bottle_detector --ros-args \
        -p image_topic:=/gripper_camera/image_raw -p info_topic:=/gripper_camera/camera_info

Publishes
  /gripper_bottle/point  geometry_msgs/PointStamped  the bottle's axis, in the camera's
                         optical frame (header.frame_id, usually gripper_camera_color_optical_frame).
                         Use TF to get it in base_link. Only published when a bottle is seen.
  /gripper_bottle/image  sensor_msgs/Image           the camera image with the box and distance drawn on.

Distance comes from the bottle's width in the image (see bottle_geometry.py),
not from the depth image: depth cameras see clear plastic badly. Set the
`diameter` parameter (m) to your bottle's; a 10% error there is a 10% error in
distance. Moving the robot is up to whoever subscribes (handoff_ros --camera).

Needs: pip install ultralytics (YOLO; the yolov8n.pt weights download on first run).
"""
import sys
import time

import cv2

from water_bottle_robot.perception.bottle_geometry import Intrinsics, box_to_point, pick_bottle, z_axis_of

COCO_BOTTLE = 39


class YoloBottles:
    """Bottle boxes [(x1, y1, x2, y2, score)] from a BGR image."""

    def __init__(self, model="yolov8n.pt", conf=0.3):
        from ultralytics import YOLO   # only needed when the node really runs
        self.yolo, self.conf = YOLO(model), conf

    def __call__(self, bgr):
        res = self.yolo(bgr, classes=[COCO_BOTTLE], conf=self.conf, verbose=False)[0]
        return [(*map(float, b), float(s)) for b, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]


def process(bgr, K, detect, diameter, previous=None, up=None):
    """One frame: (box, point) for the tracked bottle, or (None, None).
    up: world-up in the camera frame (corrects for camera tilt, see bottle_geometry)."""
    box = pick_bottle(detect(bgr), previous)
    if box is None:
        return None, None
    return box, box_to_point(box, K, diameter, up)


def annotate(bgr, box, point):
    img = bgr.copy()
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        color = (0, 200, 0) if point is not None else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = (f"bottle {box[4]:.2f}  {point[2] * 100:.0f} cm" if point is not None
                 else f"bottle {box[4]:.2f} (cut off: no distance)")
        cv2.putText(img, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    else:
        cv2.putText(img, "no bottle", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return img


def create_node(detect=None):
    """The node (rclpy must be initialized). detect: bottle detector, default YOLO."""
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PointStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    class GripperBottleDetector(Node):
        def __init__(self, detect):
            super().__init__("gripper_bottle_detector")
            p = self.declare_parameter
            self.image_topic = p("image_topic", "/gripper_camera/color/image_rect_raw").value
            self.info_topic = p("info_topic", "/gripper_camera/color/camera_info").value
            self.diameter = p("diameter", 0.07).value
            self.up_frame = p("up_frame", "base_link").value   # its +z is "up" (TF; corrects camera tilt)
            self.max_rate = p("max_rate", 5.0).value
            model, conf = p("model", "yolov8n.pt").value, p("conf", 0.3).value
            self.detect = detect or YoloBottles(model, conf)
            self.bridge = CvBridge()
            import tf2_ros
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.K, self.prev, self.last_t, self.last_report = None, None, 0.0, 0.0
            self.seen = 0
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)   # newest frame only
            self.create_subscription(CameraInfo, self.info_topic, self.on_info, qos)
            self.create_subscription(Image, self.image_topic, self.on_image, qos)
            self.pub_point = self.create_publisher(PointStamped, "/gripper_bottle/point", 10)
            self.pub_image = self.create_publisher(Image, "/gripper_bottle/image", 1)
            self.get_logger().info(f"watching {self.image_topic} (bottle diameter {self.diameter} m)")

        def up_in(self, frame):
            """up_frame's +z in the camera frame, or None (no TF yet: no tilt correction)."""
            import rclpy.time
            try:
                q = self.tf_buffer.lookup_transform(frame, self.up_frame, rclpy.time.Time()).transform.rotation
            except Exception:
                self.get_logger().warn(f"no TF {frame} <- {self.up_frame}: distance not tilt-corrected",
                                       throttle_duration_sec=5.0)
                return None
            return z_axis_of(q.x, q.y, q.z, q.w)

        def on_info(self, msg):
            self.K = Intrinsics.from_camera_info(msg)

        def on_image(self, msg):
            now = time.monotonic()
            if self.K is None or now - self.last_t < 1.0 / self.max_rate:
                return
            self.last_t = now
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            box, point = process(bgr, self.K, self.detect, self.diameter, self.prev, self.up_in(msg.header.frame_id))
            self.prev = box
            if point is not None:
                out = PointStamped()
                out.header = msg.header
                out.point.x, out.point.y, out.point.z = point
                self.pub_point.publish(out)
                self.seen += 1
            img = self.bridge.cv2_to_imgmsg(annotate(bgr, box, point), "bgr8")
            img.header = msg.header
            self.pub_image.publish(img)
            if now - self.last_report > 2.0:
                self.last_report = now
                self.get_logger().info(f"bottle at {point[2] * 100:.0f} cm" if point is not None else "no bottle")

    return GripperBottleDetector(detect)


def main(argv=None):
    import rclpy
    rclpy.init(args=sys.argv if argv is None else argv)
    node = create_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
