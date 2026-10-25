import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path("MotionWatch").resolve()))
from detector import YOLODetector
from tracker import KinematicTracker
from theft_analyzer import TheftAnalyzer

cap = cv2.VideoCapture("MotionWatch/Mynewsdesk.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
detector = YOLODetector()
tracker = KinematicTracker()
analyzer = TheftAnalyzer(confidence_threshold=0.40)

print(f"[Diagnose] Processing Mynewsdesk.mp4 (FPS: {fps:.1f})...", flush=True)

frame_idx = 0
alerts = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h0, w0 = frame.shape[:2]
    if w0 > 960:
        frame = cv2.resize(frame, (960, int(h0 * 960 / w0)))
    dets = detector.detect_and_track(frame, persist=True)
    tracks = tracker.update(dets, frame_idx, fps)
    evt = analyzer.analyze_frame(frame, tracks, frame_idx, fps)
    if evt:
        alerts.append((frame_idx, evt))
        print(f"\n*** FRAME {frame_idx}: AUTOMATIC THEFT DETECTED! ***", flush=True)
        print(f"    Event ID: {evt['event_id']}", flush=True)
        print(f"    Confidence: {evt['confidence']*100:.1f}%", flush=True)
        print(f"    Details: {evt['details']}", flush=True)

cap.release()
print(f"\n[Diagnose] Finished. Total automatic alerts: {len(alerts)}", flush=True)
