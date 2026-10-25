"""
MotionWatch: Person/vehicle detection + restricted-zone intrusion alerts.

Features:
- YOLO detection for people and common road vehicles
- ByteTrack IDs through Ultralytics tracking
- Configurable restricted polygon (default: central rectangle)
- Alarm sound, on-screen alert, screenshot, CSV event log
- OpenCV display windows

Important:
This detects a person entering a configured zone. It does NOT determine whether
a theft or crime has occurred.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
from ultralytics import YOLO

# -------------------- SETTINGS --------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
ALERTS_DIR = OUTPUT_DIR / "alerts"
OUTPUT_DIR.mkdir(exist_ok=True)
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov8n.pt"  # Downloads automatically on first run
CONFIDENCE = 0.35
RESIZE_WIDTH = 960
ALERT_COOLDOWN_SECONDS = 3.0

# COCO class IDs: person=0, car=2, motorcycle=3, bus=5, truck=7
DETECT_CLASSES = [0, 2, 3, 5, 7]
PERSON_CLASS_ID = 0

# Restricted zone is set as fractions of frame width/height.
# Change these values after looking at your video:
# x from 0.35 to 0.65, y from 0.25 to 0.90.
ZONE_X1, ZONE_Y1 = 0.35, 0.25
ZONE_X2, ZONE_Y2 = 0.65, 0.90

# --------------------------------------------------


def choose_video():
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    selected = filedialog.askopenfilename(
        title="Select a surveillance video for MotionWatch",
        initialdir=str(PROJECT_DIR),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return selected


def play_alarm():
    """Short local alarm; uses Windows sound when available."""
    try:
        import winsound
        winsound.Beep(1200, 350)
        winsound.Beep(850, 350)
    except (ImportError, RuntimeError):
        # Terminal bell fallback on non-Windows systems.
        print("\a", end="", flush=True)


def main():
    video_path = choose_video()
    if not video_path:
        print("No video selected. Exiting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    video_name = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(MODEL_NAME)

    csv_path = OUTPUT_DIR / f"{video_name}_alerts.csv"
    event_count = 0
    frame_number = 0
    last_alarm_time = 0.0
    # Track IDs currently inside the zone; used to detect entry transitions.
    persons_inside_zone = set()
    event_rows = []
    start_time = time.time()
    writer = None

    print("\nMotionWatch started.")
    print("Video:", video_path)
    print("Alert rule: person enters the restricted zone.")
    print("Press Q in the video window to stop.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1

            h0, w0 = frame.shape[:2]
            if w0 > RESIZE_WIDTH:
                new_h = int(h0 * RESIZE_WIDTH / w0)
                frame = cv2.resize(frame, (RESIZE_WIDTH, new_h))
            height, width = frame.shape[:2]

            # Restricted rectangular polygon, scaled to current frame.
            x1, y1 = int(width * ZONE_X1), int(height * ZONE_Y1)
            x2, y2 = int(width * ZONE_X2), int(height * ZONE_Y2)
            zone = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            zone_contour = __import__("numpy").array(zone, dtype="int32")

            # Run object detection + tracking. Persist keeps track identities
            # across frames. Only selected COCO classes are displayed.
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=DETECT_CLASSES,
                conf=CONFIDENCE,
                verbose=False,
            )[0]

            current_person_ids_inside = set()
            alert_this_frame = False

            cv2.polylines(frame, [zone_contour], True, (0, 165, 255), 2)
            cv2.putText(
                frame, "RESTRICTED ZONE", (x1 + 5, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
            )

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [-1] * len(xyxy)

                for box, class_id, confidence, track_id in zip(
                    xyxy, cls_ids, confs, track_ids
                ):
                    x1b, y1b, x2b, y2b = box.tolist()
                    label = model.names.get(int(class_id), str(class_id))
                    is_person = int(class_id) == PERSON_CLASS_ID

                    # Bottom-centre is a useful approximation of where a
                    # person's feet meet the ground.
                    center_x = (x1b + x2b) // 2
                    foot_y = y2b
                    inside = (
                        x1 <= center_x <= x2 and y1 <= foot_y <= y2
                    )

                    color = (0, 255, 0)
                    if is_person and inside:
                        color = (0, 0, 255)
                        if track_id >= 0:
                            current_person_ids_inside.add(int(track_id))
                            if int(track_id) not in persons_inside_zone:
                                alert_this_frame = True
                                event_count += 1
                                timestamp = frame_number / fps
                                now_label = datetime.now().strftime("%Y%m%d_%H%M%S")
                                screenshot_path = ALERTS_DIR / (
                                    f"{video_name}_intrusion_{event_count}_{now_label}.jpg"
                                )
                                cv2.imwrite(str(screenshot_path), frame)
                                event_rows.append([
                                    frame_number,
                                    round(timestamp, 2),
                                    int(track_id),
                                    "PERSON_ENTERED_RESTRICTED_ZONE",
                                    round(float(confidence), 3),
                                    str(screenshot_path),
                                ])
                                print(
                                    f"ALERT #{event_count}: person ID {track_id} "
                                    f"entered zone at {timestamp:.2f}s"
                                )

                    text_label = f"{label} {confidence:.2f}"
                    if track_id >= 0:
                        text_label += f" ID:{track_id}"
                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 2)
                    cv2.putText(
                        frame, text_label, (x1b, max(20, y1b - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

            # If a person has just entered the zone, show a visible warning.
            if alert_this_frame:
                cv2.putText(
                    frame, "INTRUSION ALERT!", (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3
                )
                now = time.time()
                if now - last_alarm_time >= ALERT_COOLDOWN_SECONDS:
                    play_alarm()
                    last_alarm_time = time.time()

            persons_inside_zone = current_person_ids_inside

            # HUD
            cv2.rectangle(frame, (0, 0), (310, 85), (25, 25, 25), -1)
            cv2.putText(
                frame, "MOTIONWATCH", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                frame, f"Alerts: {event_count}", (12, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            elapsed = max(time.time() - start_time, 0.001)
            cv2.putText(
                frame, f"FPS: {frame_number / elapsed:.1f}", (12, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            if writer is None:
                output_video = OUTPUT_DIR / f"{video_name}_motionwatch.mp4"
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    min(max(fps, 1), 30),
                    (width, height),
                )
                if not writer.isOpened():
                    print("WARNING: Could not create output video; display will continue.")
                    writer = None

            if writer is not None:
                writer.write(frame)

            cv2.imshow("MotionWatch - Detection and Restricted-Zone Alerts", frame)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "Frame", "Timestamp_Seconds", "Track_ID", "Event",
                "Confidence", "Screenshot_Path"
            ])
            csv_writer.writerows(event_rows)

        print("\n========== MOTIONWATCH REPORT ==========")
        print("Frames processed:", frame_number)
        print("Restricted-zone entry alerts:", event_count)
        print("Alert CSV:", csv_path)
        print("Alert screenshots:", ALERTS_DIR)
        print("Output folder:", OUTPUT_DIR)
        print("========================================")


if __name__ == "__main__":
    main()
