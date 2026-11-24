# Gripper camera: line up with the bottle

The fixed-spot handoff (`handoff_ros`) reaches to a spot you measure. With
`--camera`, it also looks at the bottle with the gripper camera (the D405 on the
wrist) and corrects itself before grasping, so the measured spot only has to be
within a few cm.

Two pieces:

- **`gripper_bottle_detector`** (a separate node): runs YOLO on the gripper
  camera and publishes where the bottle is (`/gripper_bottle/point`, in the
  camera's frame) plus an annotated image (`/gripper_bottle/image`). It never
  moves the robot.
- **`handoff_ros --camera`**: adds an `ALIGN` step between unfolding and
  reaching. With the arm 15 cm short of the planned grasp, it takes the median
  of 5 camera estimates, converts them to the base frame with TF, then:
  - drives the base straight **forward/back** to put the bottle on the arm's
    line (the side-to-side error in the camera image);
  - changes **how far the arm reaches** (the bottle's distance).

  It repeats until it's within 1 cm. Corrections are capped at 8 cm (base)
  and 10 cm (reach): a bigger miss means the spot or the detection is wrong,
  so it pulls the arm back and stops instead of chasing it. No detection
  does the same. The height still comes from `--hand-top`.

## How it measures distance

Depth cameras see clear plastic badly, so the distance comes from the
bottle's **width in the image**: the box's left and right edges just touch the
bottle, and with its diameter known (`diameter` parameter, default 7 cm) that
gives the distance. A 10% wrong diameter is a 10% wrong distance (3 cm at 30
cm), so set it to your bottle's.

The camera tilts down about 10 degrees, which makes the bottle's near end look
widest. The node corrects for that using TF (which way is "up" in the camera).
Checked against the robot model's D405 in MuJoCo (`test/test_gripper_camera.py`),
with the arm posed the way `ALIGN` looks: within 4 mm both ways, for bottles 0-4
cm off the line. What that can't check is how tight YOLO's box is on a real clear
bottle. A box 3 px too wide reads about 1 cm too close.

## On the robot

Once: `pip3 install ultralytics` (YOLO; `yolov8n.pt` downloads on first run).
Then rebuild as usual (`cd ~/ament_ws && colcon build --packages-select water_bottle_robot`).

Four terminals (each `source ~/ament_ws/install/setup.bash`):

```bash
ros2 launch stretch_core stretch_driver.launch.py mode:=position   # 1: driver (also publishes TF)
ros2 launch stretch_core d405_basic.launch.py                      # 2: gripper camera
ros2 run water_bottle_robot gripper_bottle_detector                # 3: detector (-p diameter:=0.075 if needed)
ros2 run rqt_image_view rqt_image_view /gripper_bottle/image       # 4: watch it (robot screen)
```

### Step 1: check the detector (nothing moves)

Hold the bottle in front of the gripper camera, wherever it's pointing, 20-35
cm away. The image should show a green box and a distance. Compare it with a
tape measure from the camera's lens to the bottle's center: it should be
within about 1 cm. If it's consistently off, check the `diameter` parameter.
An orange box means the bottle is cut off by the image edge, so no distance.

### Step 2: dry run with the camera

```bash
ros2 run water_bottle_robot handoff_ros --distance D --hand-top H --dry-run --camera
```

Put the bottle in its stand **3-4 cm in front of the cross** on purpose. After
unfolding, the robot reaches part way and prints:

```
ALIGN  looking at the bottle with the gripper camera
SEEN   bottle +3.5 cm off the arm line, +0.4 cm from the planned reach, at height 0.93 m
SEEN   bottle +0.3 cm off the arm line, ...
ALIGNED base moved +3.5 cm, arm 0.335 -> 0.339
```

The base drives forward about 3.5 cm, then the open gripper reaches around the
bottle. If the base moves the **wrong way**, stop (run-stop) and tell me: that
means TF or the image orientation differs from the robot model.

### Step 3: the handoff with the camera

```bash
ros2 run water_bottle_robot handoff_ros --distance D --hand-top H --camera
```

The rest is the same as without the camera (docs/handoff_real_robot.md).

## In the sim

The MuJoCo driver publishes the gripper camera under different names:

```bash
ros2 run water_bottle_robot gripper_bottle_detector --ros-args \
    -p image_topic:=/gripper_camera/image_raw -p info_topic:=/gripper_camera/camera_info
```

The empty sim scene has no bottle, so there `ALIGN` only shows that the robot
stops safely ("the gripper camera doesn't see the bottle").

## Not tested yet

Everything here was tested with fakes and the MuJoCo camera model, not with
YOLO on real images:

- YOLO finding a **clear** bottle 20-35 cm from the gripper camera, partly
  covered by the person's hand.
- How tight YOLO's boxes are on clear plastic, which sets the distance accuracy.
- YOLO's speed on the robot's computer (the node runs at up to 5 Hz).
- The base correction through the real driver.
