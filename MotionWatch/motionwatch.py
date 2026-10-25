
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import csv
import time

# ==========================================
# MOTIONWATCH: SMART VIDEO MOTION ANALYSIS
# ==========================================

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_OBJECT_AREA = 700
MAX_MATCH_DISTANCE = 80
MAX_MISSED_FRAMES = 12

# ------------------------------------------
# 1. SELECT INPUT VIDEO
# ------------------------------------------

root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select a video for MotionWatch",
    initialdir=str(PROJECT_DIR),
    filetypes=[
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not video_path:
    print("No video selected. Exiting.")
    raise SystemExit

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open the selected video.")
    raise SystemExit

video_name = Path(video_path).stem

fps_source = cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(fps_source) or fps_source <= 0:
    fps_source = 25.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output files
output_video = OUTPUT_DIR / f"{video_name}_motionwatch.mp4"
output_csv = OUTPUT_DIR / f"{video_name}_events.csv"

# ------------------------------------------
# 2. BACKGROUND SUBTRACTION
# ------------------------------------------

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=35,
    detectShadows=True
)

kernel = np.ones((5, 5), dtype=np.uint8)

# ------------------------------------------
# 3. OBJECT TRACKING
# ------------------------------------------

tracks = {}
next_track_id = 1

def update_tracks(detections):
    """Match detected object centres with existing tracks."""
    global next_track_id

    for track in tracks.values():
        track["missed"] += 1

    unmatched_tracks = set(tracks.keys())
    detection_ids = []

    # Match each detection to the nearest unmatched track
    for point in detections:
        best_id = None
        best_distance = MAX_MATCH_DISTANCE

        for track_id in unmatched_tracks:
            old_point = tracks[track_id]["center"]

            distance = np.linalg.norm(
                np.array(point, dtype=float)
                - np.array(old_point, dtype=float)
            )

            if distance < best_distance:
                best_distance = distance
                best_id = track_id

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1

            tracks[best_id] = {
                "center": point,
                "missed": 0,
                "trail": [],
                "inside_zone": False
            }
        else:
            unmatched_tracks.remove(best_id)

        track = tracks[best_id]
        track["center"] = point
        track["missed"] = 0
        track["trail"].append(point)
        track["trail"] = track["trail"][-30:]

        detection_ids.append(best_id)

    # Remove tracks missing for too long
    for track_id in list(tracks.keys()):
        if tracks[track_id]["missed"] > MAX_MISSED_FRAMES:
            del tracks[track_id]

    return detection_ids


# ------------------------------------------
# 4. VIDEO OUTPUT INITIALIZATION
# ------------------------------------------

writer = None
frame_number = 0
event_count = 0
processing_start = time.time()
previous_frame_time = time.time()

fps_display = 0.0
event_records = []

print("\nMotionWatch started!")
print("Input:", video_path)
print("Press Q in the video window to stop.\n")

# ------------------------------------------
# 5. PROCESS EACH VIDEO FRAME
# ------------------------------------------

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Resize large frames for faster processing
    height, width = frame.shape[:2]

    if width > 960:
        new_height = int(height * 960 / width)
        frame = cv2.resize(frame, (960, new_height))

    height, width = frame.shape[:2]

    # Define the central monitoring zone
    zone_left = int(width * 0.35)
    zone_right = int(width * 0.65)

    # --------------------------------------
    # A. IMAGE PREPROCESSING
    # --------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --------------------------------------
    # B. MOTION MASK
    # --------------------------------------

    mask = background_subtractor.apply(blurred)

    # Remove shadows (MOG2 marks shadows approximately as 127)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological noise removal
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    # --------------------------------------
    # C. CONTOUR-BASED OBJECT DETECTION
    # --------------------------------------

    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_OBJECT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        center_x = x + w // 2
        center_y = y + h // 2

        detections.append((center_x, center_y))
        boxes.append((x, y, w, h, area))

    # --------------------------------------
    # D. TRACK OBJECTS
    # --------------------------------------

    detection_ids = update_tracks(detections)

    active_objects = len(detections)
    zone_alert_this_frame = False

    # Draw the monitoring zone
    cv2.rectangle(
        frame,
        (zone_left, 0),
        (zone_right, height - 1),
        (255, 180, 0),
        2
    )

    cv2.putText(
        frame,
        "MONITORING ZONE",
        (zone_left + 5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2
    )

    # Draw object boxes, IDs and trails
    for index, box in enumerate(boxes):
        x, y, w, h, area = box
        center_x, center_y = detections[index]
        track_id = detection_ids[index]

        track = tracks[track_id]

        # Determine whether the object is inside the zone
        inside_zone = zone_left <= center_x <= zone_right

        # Record entry when the object moves from outside to inside
        if inside_zone and not track["inside_zone"]:
            event_count += 1
            zone_alert_this_frame = True

            event_records.append([
                frame_number,
                round(frame_number / fps_source, 2),
                track_id,
                "ZONE_ENTRY",
                round(float(area), 1)
            ])

        track["inside_zone"] = inside_zone

        box_color = (0, 0, 255) if inside_zone else (0, 255, 0)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (255, 0, 255),
            -1
        )

        # Draw motion trail
        trail = track["trail"]

        for j in range(1, len(trail)):
            cv2.line(
                frame,
                trail[j - 1],
                trail[j],
                (255, 0, 255),
                2
            )

    # --------------------------------------
    # E. PERFORMANCE STATISTICS
    # --------------------------------------

    current_time = time.time()
    elapsed = current_time - previous_frame_time

    if elapsed > 0:
        instantaneous_fps = 1.0 / elapsed

        # Smooth the displayed FPS
        fps_display = (
            0.8 * fps_display + 0.2 * instantaneous_fps
            if fps_display > 0
            else instantaneous_fps
        )

    previous_frame_time = current_time

    progress = (
        100 * frame_number / total_frames
        if total_frames > 0
        else 0
    )

    # Dashboard
    cv2.rectangle(frame, (0, 0), (300, 145), (25, 25, 25), -1)

    cv2.putText(
        frame,
        "MOTIONWATCH",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Moving regions: {active_objects}",
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Active tracks: {len(tracks)}",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Processing FPS: {fps_display:.1f}",
        (12, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Progress: {progress:.1f}%",
        (12, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    if zone_alert_this_frame:
        cv2.putText(
            frame,
            "ZONE ENTRY DETECTED!",
            (max(10, width - 350), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------
    # F. SAVE ANNOTATED VIDEO
    # --------------------------------------

    if writer is None:
        output_fps = min(max(fps_source, 1), 30)

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height)
        )

        if not writer.isOpened():
            print("Error: Could not create output video.")
            break

    writer.write(frame)

    # Display results
    cv2.imshow("MotionWatch - Detection and Tracking", frame)
    cv2.imshow("MotionWatch - Foreground Mask", mask)

    if cv2.waitKey(max(1, int(1000 / fps_source))) & 0xFF == ord("q"):
        break

# ------------------------------------------
# 6. SAVE EVENT LOG AND CLEAN UP
# ------------------------------------------

cap.release()

if writer is not None:
    writer.release() 

cv2.destroyAllWindows()

with open(output_csv, "w", newline="", encoding="utf-8") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "Frame",
        "Timestamp_Seconds",
        "Track_ID",
        "Event",
        "Region_Area"
    ])

    csv_writer.writerows(event_records)

elapsed_total = time.time() - processing_start

print("\n========== MOTIONWATCH REPORT ==========")
print("Video processed:", video_name)
print("Frames processed:", frame_number)
print("Zone entry events:", event_count)
print("Processing time:", round(elapsed_total, 2), "seconds")
print("Annotated video:", output_video)
print("Event log:", output_csv)
print("========================================")
