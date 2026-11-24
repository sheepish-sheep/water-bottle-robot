# Bottle handoff (MuJoCo)

Stretch takes a clear water bottle that a person is holding out, holds it
upright (the pose it would carry it to the fountain in), and hands it back
when the person tugs on it. This follows the instructor's simplification:
take the bottle from a person's hand instead of picking it up.

**Simulation only.** Nothing here sends commands to the real robot.

## Run it

```bash
scripts/run_handoff_demo.sh                      # tests, then the demo in the live viewer (VM desktop)
scripts/run_handoff_demo.sh --azimuth -150       # person standing somewhere else (deg; -90 = robot's right)
scripts/run_handoff_demo.sh --random --seed 7    # a randomized person and bottle

python3 manipulation/handoff_sim.py --trials 20 --seed 1   # randomized pass/fail batch (headless)
python3 manipulation/handoff_sim.py --video out.mp4         # render a video instead of the viewer
python3 -m pytest test/test_handoff.py -q                  # planning tests (fast)
python3 -m pytest test/test_handoff.py -q -m slow          # + full sim handoffs (~2 min)
```

No ROS needed; it uses `stretch_mujoco`'s robot model directly.

## Files

| File | What |
|---|---|
| `handoff_controller.py` | The behavior: planning math + the state machine. No MuJoCo or ROS: it talks to a small `RobotInterface` using only motions the real Stretch has (base rotate, lift, arm, wrist yaw/pitch/roll, gripper) at modest speeds. A real-robot version would implement the same interface on `stretch_body` / `stretch_core`. |
| `handoff_scene.py` | The MuJoCo scene: Stretch, a clear bottle (cylinder + shoulder + cap; any ordinary bottle shape, no markings needed), and a person whose hand holds the bottle (a weld that turns off when they let go). |
| `handoff_sim.py` | The sim robot (trapezoidal joint profiles, base heading loop), a perception stand-in (true pose + 1 cm noise), the scripted person, the checks, batches and video. |

## What it does (state machine)

```
LOCATE -> FACE -> REFINE -> RAISE -> UNFOLD -> REACH -> GRASP -> TAKE -> RETRACT -> HOLD
                                                                                     |
                     DONE <- STOW <- RELEASE <- (person tugs) <- GIVE  <-------------+
```

1. **LOCATE / FACE**: rough bottle estimate, rotate the base so the bottle is on the arm's line (the arm reaches out to the robot's right).
2. **REFINE**: second estimate from the new viewpoint. Grasp height = top of the person's hand + 7 cm; if that doesn't fit below the bottle's shoulder, refuse and ask them to hold it lower. Refuse too if the bottle is out of reach.
3. **RAISE / UNFOLD**: lift (no higher than 0.75 m yet), swing the gripper out, *then* open it, then lift the rest of the way.
4. **REACH**: fast to 8 cm short of the bottle, then slowly (4 cm/s) to the grasp.
5. **GRASP**: close; success = the fingers stalled partway (holding something), not fully closed.
6. **TAKE**: hold still for 1.8 s while the person lets go.
7. **RETRACT / HOLD**: pull the arm in with the bottle upright.
8. **GIVE / RELEASE**: extend back out, wait until the arm has settled, then wait for the person to grip the bottle and pull. Only a sustained pull (arm effort change > max(2.5 N, 5x noise) for 0.25 s) opens the gripper. If nobody pulls within 10 s, the robot keeps the bottle instead of dropping it.
9. **STOW**: close the gripper, fold it back, lower.

## Design decisions (each one came from a failure in sim)

| Decision | Why |
|---|---|
| Insert the bottle **5 cm past** the gripper's grasp point | Shallower, the fingertips pinch the bottle's near side and squeeze it out: it pivoted up to 22 deg. A sweep showed 3.5-7 cm wraps the fingers around its far side (<6 deg); 5 cm is mid-window (+-1.5 cm for vision error). |
| Grasp height from the **person's hand** | The fingertips reach 4.7 cm below the grasp point; grasping at a fixed spot on the bottle let them hit the person's palm. |
| Don't lift while the person still holds it | Lifting the top while they grip the bottom twisted the bottle +5 deg. |
| Unfold at <= 0.75 m lift, open only once unfolded, close before folding | Unfolding high hit the robot's own head; opening while tucked hit the mast. |
| Tug-to-release, armed only after the arm settles | A timer released the bottle before the person had it; a fixed effort threshold mistook a heavy bottle's swaying for a tug and dropped a 0.7 kg bottle. |
| Minimum reach 0.155 m, max 0.48 m | Unfolding sweeps the fingertips 3.5 cm past the grasp point; the arm stalled near its 0.52 m limit. |

## Results (randomized, final code)

Each trial randomizes: where the person stands (any direction, 0.54-0.83 m
away, grasp height 0.65-1.10 m), bottle diameter 6.4-8 cm, height, mass
0.03-0.8 kg (empty to full), friction 0.4-1.0, grip force 15-40 N, hand tremor
0-1 cm, where the hand holds the bottle, how quickly the person lets go / takes
it back, plus 1 cm vision noise on every estimate.

Checks per trial: finished; never touched the person; no self-collision;
only the gripper touched the bottle; never dropped; tilt < 10 deg and slip
< 2 cm while held; handed back; no motion timeouts.

| Batch | Trials | Full pass | Safe refusal ("hold it lower") | Safety failure (touched person / self-collision / dropped) | Other failure |
|---|---|---|---|---|---|
| Final code, seeds 9-12 | 32 | **30** | 1 | **0** | 1: tilt 10.4 deg (hand tremor 10 mm) |
| One revision earlier\*, seeds 5-8 | 60 | 53 | 3 | **0** | 3 tilt 10.5-15.6 deg, 1 of them also slipped 2.5 cm (0.6 kg, friction 0.49); 1 arm stalled near full extension |

\* Before the max-reach limit (which fixed that stall) and a cosmetic change to the
person's arm motion. Earlier versions scored 20/30 and 35/42; every failure
type found along the way is in the design-decisions table above.

In passing trials: tilt median 4-5 deg (worst 9.2), slip median 0.5 cm (worst
1.7), about 53 s per handoff. Remaining failures are all the bottle tilting or
slipping in the grip, never anything touching the person.

Raw results: `~/ros2_ws/artifacts/handoff/{final,confirm}_seed*.json`.

## What this does NOT show (test these on the real robot)

Simulation can't settle these. In rough order of risk:

1. **Vision.** The sim "perception" is the true pose + 1 cm noise. The real
   pipeline (YOLO bottle box + MediaPipe hand landmarks + depth taken from the
   *hand*, because depth cameras see clear plastic badly) has to hit about
   +-1.5 cm along the approach direction; at 2-3 cm error the sim already
   starts failing. Measure it on recorded head-camera data before trusting it.
2. **Grip force and friction.** The real SG3 gripper's squeeze is unknown here;
   the sim caps it at 15-40 N. A full (0.6-0.8 kg), slippery (wet, ~0.4
   friction) bottle with a weak grip slid 2.5 cm in sim. A full bottle with
   condensation is exactly this case: check slip on the real gripper.
3. **Tug detection.** The 2.5 N threshold is a sim number. Record the real arm
   effort while holding still and while someone pulls, and set the threshold
   from that.
4. **Tilt from the person.** While both hold the bottle, a trembling hand rocks
   it in the fingers (up to ~15 deg in the worst sim case) and it stays tilted
   after they let go. Fine for carrying; may matter under the fountain.
5. **Person safety.** The robot moves slowly and the sim checks contact, but it
   doesn't watch the person during the reach. On hardware: someone on the
   run-stop, slow speeds first, the person holds still.
6. **Bottle deformation.** The sim bottle is rigid; a thin plastic bottle
   squeezed at 30-40 N will dent.
