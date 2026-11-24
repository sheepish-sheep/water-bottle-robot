# Bottle handoff on the real Stretch (fixed-spot version)

The same handoff as the simulation (raise, unfold, reach, grasp, wait for the
person to let go, hold, hand back on a tug, stow), but **without the camera**:
the person holds the bottle at a spot you measure and mark beforehand, and you
tell the robot where it is.

- `handoff_real.py`: the only program here that **moves the robot**. It asks
  before moving (type `go`), runs at half the sim speed, stops on Ctrl-C, on
  the run-stop button, or if any motion doesn't finish in time.
- `handoff_real_check.py`: read-only. It never moves the robot.

Both run **on the robot's own computer** and use Hello Robot's `stretch_body`
library, which is already installed there. Nothing here uses ROS.

> This code has only been tested against the MuJoCo simulation and a fake
> robot (`test/test_handoff_real.py`), never on hardware. Go through the
> stages in order and don't skip the dry runs.

## What you need

- The robot, charged, in a clear area: about 1.5 m free on its **right** side
  (the arm reaches out to the right).
- Tape measure, floor tape.
- A dry, ordinary water bottle, 6.4-8 cm across. Start with an empty one.
- For the first tests: something to hold the bottle still at a fixed height,
  e.g. a bottle-sized cup or clamp on a stool or a stack of boxes.
- Two people: one at the laptop, one with a hand on the **run-stop** (the lit
  button on top of the robot's head).

## 0. Get the code onto the robot (once, then `git pull`)

```bash
ssh <user>@<robot>
cd ~/ament_ws/src
git clone https://github.com/sheepish-sheep/water-bottle-robot.git
cd water-bottle-robot && git checkout handoff-demo
```

Later updates: `cd ~/ament_ws/src/water-bottle-robot && git pull`.

All commands below run on the robot, from `~/ament_ws/src/water-bottle-robot`.

## 1. Every power-up

```bash
stretch_free_robot_process.py      # make sure nothing else (stretch_driver/ROS) is using the robot
stretch_robot_home.py              # homing: MOVES the lift, arm and gripper to find their limits
python3 manipulation/handoff_real_check.py status   # read-only
```

`status` should say `homed: True`, `run-stop: released`, a battery above
about 11.5 V, and list `wrist_yaw ... stretch_gripper` under end of arm.

## 2. Mark the spot

- **Distance:** measure sideways, straight out to the robot's **right**, from
  the middle of the base (halfway between the two drive wheels) to where the
  center of the bottle will be. Use **0.70 m** to start (works for 0.53-0.84).
  Put a tape cross on the floor there.
- **Hand height:** height above the floor of the **top of the hand** (or the
  top of the stand) holding the bottle. Use **0.83 m** to start. The robot
  grasps 7 cm above it, so at least 15 cm of bottle must stick out above the
  hand.

## 3. Dry run: line up the spot (no bottle needed)

```bash
python3 manipulation/handoff_real.py --distance 0.70 --hand-top 0.83 --dry-run
```

The robot raises, unfolds the gripper, opens it, reaches slowly to the grasp
point, holds 4 s, then comes back and stows. Nothing is grasped.

While it holds: the spot between the open fingers, about **5 cm in from the
fingertips**, is where the bottle's center should be. Compare with your tape
cross:

| Gripper ends up... | Fix |
|---|---|
| short of / past the cross (sideways) | change `--distance` by that much |
| too high / too low | change `--hand-top` by that much |
| in front of / behind the cross | move the cross (the bottle must be on the arm's line) |

Repeat until it lines up within about 1 cm. Then do it once more with the
bottle in its stand at the cross: the open fingers should end up around the
bottle without touching it.

## 4. Measure the tug threshold

The robot hands the bottle back only when the person pulls on it. How much the
arm's motor effort changes for "a gentle pull" has to be measured on this
robot:

```bash
python3 manipulation/handoff_real_check.py tug      # read-only
```

It needs the arm out about 20 cm with the gripper unfolded first (put it there
with the gamepad teleop, then quit that). It records the effort while nobody
touches the robot, then while someone pulls the gripper gently away from the
robot. It prints something like `--tug-threshold 6.5 --tug-sign 1`.

If it says the pulls barely changed anything, measure during a real handoff
instead:

```bash
python3 manipulation/handoff_real.py --distance D --hand-top H --tug-threshold 1000   # never lets go
#   when it offers the bottle back: wait 3 s, then pull gently 2-3 times.
#   It keeps the bottle and asks you to take it.
python3 manipulation/handoff_real_check.py tug-log ~/ament_ws/artifacts/handoff_real/run_<time>.csv
```

## 5. Handoff from the stand, then from a person

```bash
python3 manipulation/handoff_real.py --distance D --hand-top H --tug-threshold T --tug-sign S
```

with D and H from step 3 and T and S from step 4.

1. **Bottle in the stand.** When the fingers close, the robot waits 1.8 s
   for the person to let go, then pulls the arm in sideways. It doesn't lift
   first, so in those 1.8 s **lower the stand away** from under the bottle.
   That's the stand "letting go"; otherwise the arm drags the stand along.
   The robot holds the bottle, then reaches back out. When it's offered back,
   hold the bottle and pull gently: it opens and backs away.
2. **Empty bottle, held by a person** standing on the cross, holding the
   bottle near the bottom with the top of their hand at H, and **holding
   still** until the fingers close. Let go once the gripper has closed.
3. **Full bottle**, the same way.

If nobody pulls within 10 s, it keeps the bottle, retracts, and asks you to take
it before opening the gripper. It never just drops it.

## Tuning

| Problem | Try |
|---|---|
| "gripper closed on nothing" | spot is off: redo the dry run |
| bottle slips in the fingers | squeeze harder: `--close-pct -70` (default -50; -100 is the maximum) |
| bottle dents | squeeze less: `--close-pct -30` |
| "closed" but it says it's not holding | `handoff_real_check.py watch` with the bottle in the closed gripper; set `--holding-pct` below that reading |
| lets go too easily / not at all | redo step 4, or raise / lower `--tug-threshold` |
| "motion didn't finish" | something blocked a joint: check for contact, then run again |
| everything feels slow | `--speed 0.8` (only after it works at 0.5) |

Every run writes a log of phases, joint positions, gripper and arm effort to
`~/ament_ws/artifacts/handoff_real/run_<time>.csv`. Send it over when something
goes wrong.

## For the video

Film from the robot's right-front, so the bottle, the gripper and the person's
hand are all visible. A full handoff takes about 1-1.5 minutes at `--speed 0.5`.
