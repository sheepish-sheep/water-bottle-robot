"""
Webcam demo: detect faces, hands, and bottles, then pick the bottle
the person is holding out ("offered bottle").

Setup (once):
    pip install opencv-python mediapipe ultralytics
    # MediaPipe model files (put next to this script):
    wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
    wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    # YOLO weights (yolov8n.pt) download automatically on first run.

Run:
    python offered_bottle_detector.py            # webcam 0
    python offered_bottle_detector.py --cam 1
    python offered_bottle_detector.py --video test.mp4

Keys: q = quit, s = save screenshot, f = toggle fullscreen
    python offered_bottle_detector.py --fullscreen   # start fullscreen
"""
import argparse
import time

import cv2
import numpy as np

COCO_BOTTLE = 39  # "bottle" class id in COCO


# ---------- pure logic (no models, easy to test) ----------

def point_to_box_dist(px, py, box):
    """Distance from a point to a box (x1,y1,x2,y2). 0 if inside."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return float(np.hypot(dx, dy))


def pick_offered_bottle(bottles, hands, max_norm_dist=0.5):
    """
    bottles: list of (x1,y1,x2,y2,score) in pixels
    hands:   list of hands, each a list of (x,y) landmark pixels
    Returns index of the bottle closest to a hand, or None.

    Distance is normalized by bottle box diagonal so a far-away small
    bottle doesn't win just because it's small.
    """
    if not bottles or not hands:
        return None
    best_i, best_d = None, float("inf")
    for i, (x1, y1, x2, y2, _) in enumerate(bottles):
        diag = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        for hand in hands:
            d = min(point_to_box_dist(px, py, (x1, y1, x2, y2)) for px, py in hand)
            d /= diag
            if d < best_d:
                best_i, best_d = i, d
    return best_i if best_d <= max_norm_dist else None


# ---------- model wrappers ----------

class Detectors:
    def __init__(self, face_model, hand_model, yolo_model, conf):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        from ultralytics import YOLO

        self.mp = mp
        self.conf = conf
        mode = vision.RunningMode.VIDEO
        self.face = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mpt.BaseOptions(model_asset_path=face_model),
                running_mode=mode,
                min_detection_confidence=0.5,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=hand_model),
                running_mode=mode,
                num_hands=2,
            )
        )
        self.yolo = YOLO(yolo_model)
        self.last_ts = -1

    def run(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        # MediaPipe VIDEO mode needs strictly increasing timestamps
        ts = int(time.monotonic() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts

        faces = []
        for d in self.face.detect_for_video(img, ts).detections:
            b = d.bounding_box
            faces.append((b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height))

        hands = []
        for lm in self.hands.detect_for_video(img, ts).hand_landmarks:
            hands.append([(p.x * w, p.y * h) for p in lm])

        bottles = []
        res = self.yolo(frame_bgr, classes=[COCO_BOTTLE], conf=self.conf, verbose=False)[0]
        for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            bottles.append((x1, y1, x2, y2, float(score)))

        return faces, hands, bottles


# ---------- drawing ----------

def draw(frame, faces, hands, bottles, offered):
    for x1, y1, x2, y2 in faces:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
        cv2.putText(frame, "face", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    for hand in hands:
        for px, py in hand:
            cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 255), -1)
    for i, (x1, y1, x2, y2, s) in enumerate(bottles):
        is_off = i == offered
        color = (0, 0, 255) if is_off else (0, 200, 0)
        label = f"OFFERED {s:.2f}" if is_off else f"bottle {s:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3 if is_off else 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_off:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            # this pixel + depth is what the grasp step needs
            cv2.putText(frame, f"target px ({cx},{cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--face-model", default="blaze_face_short_range.tflite")
    ap.add_argument("--hand-model", default="hand_landmarker.task")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="bottle confidence threshold")
    ap.add_argument("--fullscreen", action="store_true", help="start fullscreen (f toggles)")
    args = ap.parse_args()

    det = Detectors(args.face_model, args.hand_model, args.yolo, args.conf)
    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    if not cap.isOpened():
        raise SystemExit("could not open camera/video")

    win = "offered bottle"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # resizable, so it can go fullscreen
    full = args.fullscreen

    def set_full(on):
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    set_full(full)
    n_frames = n_offered = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces, hands, bottles = det.run(frame)
        offered = pick_offered_bottle(bottles, hands)
        n_frames += 1
        n_offered += offered is not None
        draw(frame, faces, hands, bottles, offered)
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("f"):
            full = not full
            set_full(full)
        if k == ord("s"):
            cv2.imwrite(f"shot_{int(time.time())}.png", frame)

    cap.release()
    cv2.destroyAllWindows()
    if n_frames:
        print(f"frames: {n_frames}, offered bottle found in {n_offered} ({100 * n_offered / n_frames:.1f}%)")


if __name__ == "__main__":
    main()
