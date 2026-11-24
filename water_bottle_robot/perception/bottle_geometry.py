"""
Where a bottle is, from its box in a camera image. Pure math: no ROS, no model.

The gripper camera's depth is unreliable on clear plastic, so distance comes
from the bottle's apparent WIDTH instead: the box's left and right edges are
the two lines from the camera that just touch the bottle, so the angle between
them and the bottle's known diameter give its distance (about fx * D / w for a
bottle w pixels wide straight ahead; the exact version below also holds off to
the side of the image). The box center row gives a rough height.

Camera optical frame (ROS convention, same as camera_info): +z out of the lens,
+x right in the image, +y down in the image.
"""
import math
from dataclasses import dataclass


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float   # principal point (px)
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, msg):
        k = msg.k
        return cls(fx=k[0], fy=k[4], cx=k[2], cy=k[5], width=msg.width, height=msg.height)


def box_to_point(box, K: Intrinsics, diameter, up=None):
    """Bottle axis point (x, y, z) in the camera optical frame, from its box (x1, y1, x2, y2)
    in pixels, or None if the box can't be used.

    x, z: exact for an upright bottle (axis along the image's y): the left/right edges
    are the tangent lines from the camera, at angles al < ar. The axis is then at
    range r / sin((ar - al) / 2) along the middle direction (al + ar) / 2.
    y: from the box center row. That's the middle of the VISIBLE part of the bottle,
    not a fixed point on it, so it's only a rough height (we don't use it to grasp).

    up: which way is up (the bottle's axis), as a unit vector in the camera optical frame
    (from TF). When the camera is tilted, the end of the bottle nearer the lens looks
    widest and sets the box width, so the plain estimate is the distance to that END;
    with `up` it's corrected to the box's middle row. Without it, a 10 deg tilt makes
    the bottle look ~1.5 cm closer than it is (measured with the MuJoCo D405 model).
    """
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    if w <= 2:
        return None
    # a box cut off by the image's left/right edge is narrower than the bottle: no distance
    margin = 2
    if x1 <= margin or x2 >= K.width - 1 - margin:
        return None
    al, ar = math.atan((x1 - K.cx) / K.fx), math.atan((x2 - K.cx) / K.fx)
    rng = (diameter / 2.0) / math.sin((ar - al) / 2.0)
    mid = (al + ar) / 2.0
    x, z = rng * math.sin(mid), rng * math.cos(mid)
    if up is not None and abs(up[1]) > 0.3:
        # half the visible length along the axis, and how much nearer its near end is
        half_len = (y2 - y1) / 2.0 * z / (K.fy * abs(up[1]))
        dz = half_len * abs(up[2])
        z += dz        # (x stays: in the MuJoCo D405 check it was exact without scaling)
    v = (y1 + y2) / 2.0
    return (x, (v - K.cy) * z / K.fy, z)


def pick_bottle(boxes, previous=None):
    """The bottle to track: closest to the previous pick if there was one (so it
    doesn't jump to another bottle), else the widest (= nearest) box.
    boxes: [(x1, y1, x2, y2, score)]."""
    if not boxes:
        return None
    if previous is not None:
        pu, pv = (previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2
        return min(boxes, key=lambda b: ((b[0] + b[2]) / 2 - pu) ** 2 + ((b[1] + b[3]) / 2 - pv) ** 2)
    return max(boxes, key=lambda b: b[2] - b[0])


def z_axis_of(x, y, z, w):
    """The z axis of a frame rotated by quaternion (x, y, z, w), i.e. the rotation matrix's
    third column. For TF's camera <- base_link rotation: which way is "up" in the camera."""
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
