"""
MotionWatch — AI-Based CCTV Theft Detection and Automated Alert System
Main Application Orchestrator and Controller.
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np

# Project modules
from config import (
    PROJECT_DIR,
    OUTPUT_DIR,
    INCIDENTS_DIR,
    ANNOTATED_DIR,
    RESIZE_WIDTH,
    COLOR_BBOX_PERSON_BGR,
    COLOR_BBOX_ITEM_BGR,
    COLOR_BBOX_SUSPECT_BGR
)
from detector import YOLODetector
from tracker import KinematicTracker
from theft_analyzer import TheftAnalyzer
from alert_manager import AlertManager
from screenshot_manager import ScreenshotManager
from logger import EventLogger, AsyncVideoWriter


class MotionWatchController:
    """Central Controller managing the video loop, analytics, alerts, and UI."""

    def __init__(self, ui_mode: bool = True):
        self.ui_mode = ui_mode
        self.ui = None
        
        # Core engines
        print("[MotionWatch] Initializing subsystems...")
        self.detector = YOLODetector()
        self.tracker = KinematicTracker()
        self.theft_analyzer = TheftAnalyzer()
        self.alert_manager = AlertManager()
        self.screenshot_manager = ScreenshotManager()
        self.logger = EventLogger()
        self.video_writer: Optional[AsyncVideoWriter] = None

        # State flags
        self.video_source: Optional[str] = None
        self.is_webcam: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_monitoring: bool = False
        self.is_paused: bool = False
        self.stop_requested: bool = False
        self.sound_enabled: bool = True
        
        # Processing thread
        self.worker_thread: Optional[threading.Thread] = None
        
        # Telemetry
        self.fps_calc = 0.0
        self.frame_counter = 0
        self.video_fps = 25.0
        self.total_video_frames = 0

    def set_ui(self, ui):
        self.ui = ui
        # Load existing history into UI table
        recent_events = self.logger.get_recent_events(limit=50)
        self.ui.load_initial_events(recent_events)

    def select_video_file(self):
        """Opens standard file picker for surveillance video selection."""
        import tkinter as tk
        from tkinter import filedialog
        
        selected = filedialog.askopenfilename(
            title="Select CCTV Surveillance Video",
            initialdir=str(PROJECT_DIR),
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if selected:
            self.load_video(selected, is_webcam=False)

    def select_webcam(self):
        """Configures webcam as input stream."""
        self.load_video("0", is_webcam=True)

    def load_video(self, source_path: str, is_webcam: bool = False):
        """Loads video stream or webcam and prepares metadata."""
        self.stop_monitoring()
        self.video_source = source_path
        self.is_webcam = is_webcam

        src_id = int(source_path) if is_webcam else source_path
        temp_cap = cv2.VideoCapture(src_id)
        if not temp_cap.isOpened():
            err_msg = f"Failed to open video source: {source_path}"
            print(f"[MotionWatch] ERROR: {err_msg}")
            if self.ui:
                from tkinter import messagebox
                messagebox.showerror("Video Error", err_msg)
            return False

        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        self.video_fps = fps
        self.total_video_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        temp_cap.release()

        display_name = "Live Webcam (Device 0)" if is_webcam else Path(source_path).name
        print(f"[MotionWatch] Loaded: {display_name} ({self.video_fps:.1f} FPS, {self.total_video_frames} frames)")
        
        if self.ui:
            self.ui.selected_video_name.set(f"{display_name} ({self.video_fps:.1f} FPS)")
            self.ui.status_text.set("READY")
        return True

    def start_monitoring(self):
        """Starts video processing in worker thread."""
        if self.is_monitoring:
            if self.is_paused:
                self.is_paused = False
                if self.ui:
                    self.ui.status_text.set("MONITORING ACTIVE")
            return

        if not self.video_source:
            if self.ui:
                from tkinter import messagebox
                messagebox.showinfo("No Source", "Please select a video file or webcam first.")
            return

        self.is_monitoring = True
        self.is_paused = False
        self.stop_requested = False
        
        # Reset analyzers and trackers
        self.tracker.reset()
        self.theft_analyzer.reset()
        self.frame_counter = 0

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()

        if self.ui:
            self.ui.status_text.set("MONITORING ACTIVE")
            self.ui.status_badge.config(bg="#10b981")

    def toggle_pause(self):
        """Pauses or resumes playback."""
        if not self.is_monitoring:
            return
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "MONITORING ACTIVE"
        print(f"[MotionWatch] Playback {status}")
        if self.ui:
            self.ui.status_text.set(status)
            self.ui.status_badge.config(bg="#f59e0b" if self.is_paused else "#10b981")

    def stop_monitoring(self):
        """Stops video stream processing."""
        self.stop_requested = True
        self.is_monitoring = False
        self.is_paused = False
        
        if self.worker_thread and self.worker_thread.is_alive() and threading.current_thread() != self.worker_thread:
            self.worker_thread.join(timeout=1.0)

        if hasattr(self, "latest_frame") and self.latest_frame is not None:
            self.screenshot_manager.flush_remaining(self.latest_frame)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.alert_manager.stop_audio_alarm()

        if self.ui:
            self.ui.status_text.set("STOPPED")
            self.ui.status_badge.config(bg="#374151")
            self.ui.hide_alert_banner()

    def acknowledge_alarm(self):
        """Acknowledge active alert."""
        self.alert_manager.acknowledge_alarm()
        if self.ui:
            self.ui.hide_alert_banner()

    def trigger_manual_test(self):
        """Triggers manual test alert."""
        # Grab active track IDs if available
        active_ids = list(self.tracker.tracks.keys())
        test_event = self.theft_analyzer.trigger_manual_test(active_ids)
        
        # Trigger alert audio/visual
        if self.sound_enabled:
            self.alert_manager.trigger_alert(test_event)

        # Trigger screenshots
        if hasattr(self, "latest_frame") and self.latest_frame is not None:
            self.screenshot_manager.trigger_incident_capture(
                event_data=test_event,
                current_frame=self.latest_frame,
                trigger_frame_idx=self.frame_counter,
                fps=self.video_fps,
                video_source=self.video_source or "Manual_Test"
            )

        # Log event
        self.logger.log_event(test_event, self.video_source or "Manual_Test")

        if self.ui:
            self.ui.show_alert_banner(test_event)
            self.ui.add_event_to_table(test_event)
            self.ui.screenshots_count_text.set(f"{self.screenshot_manager.get_screenshot_count()} Captured")

    def set_theft_threshold(self, val: float):
        self.theft_analyzer.set_threshold(val)

    def set_sound_enabled(self, enabled: bool):
        self.sound_enabled = enabled
        if not enabled:
            self.alert_manager.stop_audio_alarm()

    def open_output_folder(self):
        """Opens output incidents folder in system file manager."""
        folder = str(INCIDENTS_DIR.resolve())
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            import subprocess
            subprocess.run(["xdg-open", folder])

    def _processing_loop(self):
        """Worker thread executing YOLO detection, ByteTrack tracking, and theft analysis."""
        src_id = int(self.video_source) if self.is_webcam else self.video_source
        self.cap = cv2.VideoCapture(src_id)
        
        if not self.cap.isOpened():
            print(f"[MotionWatch] ERROR: Could not open {self.video_source}")
            self.is_monitoring = False
            return

        # Prepare optional video writer
        save_annotated = self.ui.save_video_var.get() if self.ui else False
        if save_annotated and not self.is_webcam:
            out_name = f"{Path(self.video_source).stem}_motionwatch_annotated.mp4"
            out_path = ANNOTATED_DIR / out_name
            # Grab resolution
            ret_test, frame_test = self.cap.read()
            if ret_test:
                h0, w0 = frame_test.shape[:2]
                new_h = int(h0 * RESIZE_WIDTH / w0) if w0 > RESIZE_WIDTH else h0
                new_w = RESIZE_WIDTH if w0 > RESIZE_WIDTH else w0
                self.video_writer = AsyncVideoWriter(out_path, self.video_fps, new_w, new_h)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_time = time.time()
        fps_smoothing = 0.9

        while not self.stop_requested and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.04)
                continue

            loop_start = time.time()
            ok, raw_frame = self.cap.read()
            
            if not ok:
                # Video ended or disconnected
                print("[MotionWatch] Video stream reached end of file.")
                if hasattr(self, "latest_frame") and self.latest_frame is not None:
                    self.screenshot_manager.flush_remaining(self.latest_frame)
                break

            self.frame_counter += 1

            # Resize frame for efficient real-time inference
            h0, w0 = raw_frame.shape[:2]
            if w0 > RESIZE_WIDTH:
                new_h = int(h0 * RESIZE_WIDTH / w0)
                frame = cv2.resize(raw_frame, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame = raw_frame.copy()

            self.latest_frame = frame

            # 1. Object Detection & ByteTrack
            detections = self.detector.detect_and_track(frame, persist=True)

            # 2. Kinematic History Update
            active_tracks = self.tracker.update(detections, self.frame_counter, self.video_fps)

            # 3. Theft & Temporal Interaction Analysis
            event = self.theft_analyzer.analyze_frame(frame, active_tracks, self.frame_counter, self.video_fps)
            
            # If suspected theft detected, trigger alarms & screenshots!
            if event is not None:
                if self.sound_enabled:
                    self.alert_manager.trigger_alert(event)
                
                # Multi-stage screenshot captures (before.jpg, detected.jpg, after_1s, after_3s)
                self.screenshot_manager.trigger_incident_capture(
                    event_data=event,
                    current_frame=frame,
                    trigger_frame_idx=self.frame_counter,
                    fps=self.video_fps,
                    video_source=self.video_source or "CCTV"
                )

                # CSV Event Log
                self.logger.log_event(event, self.video_source or "CCTV")

                # UI Notification
                if self.ui:
                    self.ui.root.after(0, self.ui.show_alert_banner, event)
                    self.ui.root.after(0, self.ui.add_event_to_table, event)

            # 4. Advance pending screenshots
            self.screenshot_manager.update_frame(frame, self.frame_counter, self.video_fps)

            # 5. Alert state updates (timer decay, expiry)
            self.alert_manager.update()
            if self.ui and not self.alert_manager.is_alert_active and not self.alert_manager.audio_playing:
                if self.ui.alert_banner.winfo_ismapped():
                    self.ui.root.after(0, self.ui.hide_alert_banner)

            # 6. Render Overlays & Visual Annotations
            annotated_frame = self._render_annotations(frame, detections, active_tracks)
            final_frame = self.alert_manager.render_overlay(annotated_frame, self.is_monitoring, self.fps_calc)

            # Write to video recorder if enabled
            if self.video_writer:
                self.video_writer.write_frame(final_frame)

            # Calculate FPS
            elapsed = time.time() - loop_start
            inst_fps = 1.0 / max(1e-4, elapsed)
            self.fps_calc = (fps_smoothing * self.fps_calc) + ((1.0 - fps_smoothing) * inst_fps)

            # Update UI Canvas
            if self.ui:
                self.ui.root.after(0, self.ui.update_frame, final_frame)
                
                # Telemetry updates (every 5 frames to avoid GUI event spam)
                if self.frame_counter % 5 == 0:
                    persons_count = len([t for t in active_tracks.values() if t.is_person and t.missed_frames == 0])
                    tracks_str = f"{len(active_tracks)} Tracks ({persons_count} Persons)"
                    shots_str = f"{self.screenshot_manager.get_screenshot_count()} Captured"
                    fps_str = f"{self.fps_calc:.1f} FPS"
                    
                    self.ui.root.after(0, self.ui.tracks_count_text.set, tracks_str)
                    self.ui.root.after(0, self.ui.screenshots_count_text.set, shots_str)
                    self.ui.root.after(0, self.ui.fps_text.set, fps_str)
            elif self.frame_counter % 30 == 0:
                # Headless console logging
                pct = (self.frame_counter / self.total_video_frames * 100) if self.total_video_frames > 0 else 0.0
                print(f"[MotionWatch Headless] Frame {self.frame_counter}/{self.total_video_frames} ({pct:.1f}%) | Tracks: {len(active_tracks)} | FPS: {self.fps_calc:.1f}", flush=True)

            # Rate limiting for realistic playback timing if prerecorded and in GUI mode
            if self.ui and not self.is_webcam:
                target_delay = (1.0 / self.video_fps) - elapsed
                if target_delay > 0.002:
                    time.sleep(target_delay)

        # Cleanup on finish
        self.stop_monitoring()

    def _render_annotations(self, frame: np.ndarray, detections: list, active_tracks: dict) -> np.ndarray:
        """Draws bounding boxes, tracking trails, and speed tags on frame."""
        out = frame.copy()
        suspect_id = self.alert_manager.active_event.get("suspect_id", -1) if self.alert_manager.is_alert_active else -1

        for det in detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cname = det["class_name"]

            # Highlight suspect if active alert
            if tid == suspect_id and suspect_id != -1:
                color = COLOR_BBOX_SUSPECT_BGR
                box_thickness = 3
                label = f"⚠ SUSPECT ID: {tid} ({cname})"
            elif det["is_person"]:
                color = COLOR_BBOX_PERSON_BGR
                box_thickness = 2
                speed = active_tracks[tid].current_speed if tid in active_tracks else 0.0
                items = active_tracks[tid].carried_items if tid in active_tracks else []
                items_str = f" [{','.join(items)}]" if items else ""
                label = f"ID {tid} {cname} {speed:.1f}px/f{items_str}"
            elif det["is_item"]:
                color = COLOR_BBOX_ITEM_BGR
                box_thickness = 2
                label = f"{cname} ({det['confidence']:.2f})"
            else:
                color = (200, 200, 200)
                box_thickness = 1
                label = f"{cname}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)

            # Tag label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(out, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Trajectory trail
            if tid in active_tracks:
                centers = active_tracks[tid].centers[-20:]
                for k in range(1, len(centers)):
                    pt1 = (int(centers[k - 1][0]), int(centers[k - 1][1]))
                    pt2 = (int(centers[k][0]), int(centers[k][1]))
                    cv2.line(out, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

        return out

    def shutdown(self):
        """Graceful application teardown."""
        self.stop_monitoring()


def main():
    parser = argparse.ArgumentParser(description="MotionWatch — AI-Based CCTV Theft Detection & Alert System")
    parser.add_argument("--video", type=str, help="Path to input CCTV video file")
    parser.add_argument("--webcam", action="store_true", help="Use live webcam device 0")
    parser.add_argument("--headless", action="store_true", help="Run without GUI in command-line mode")
    args = parser.parse_args()

    controller = MotionWatchController(ui_mode=not args.headless)

    # Initial video loading if provided via CLI
    if args.webcam:
        controller.load_video("0", is_webcam=True)
    elif args.video:
        controller.load_video(args.video, is_webcam=False)
    else:
        # Default to existing sample video if found
        default_samples = [
            PROJECT_DIR / "sample.mp4",
            PROJECT_DIR / "MPS Phone Theft - Media Cut - June 2026.mp4",
            PROJECT_DIR / "sample2.mp4"
        ]
        for s in default_samples:
            if s.exists():
                controller.load_video(str(s), is_webcam=False)
                break

    if args.headless:
        print("[MotionWatch] Starting in HEADLESS mode...")
        controller.start_monitoring()
        try:
            while controller.is_monitoring:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[MotionWatch] Interrupted by user.")
        finally:
            controller.shutdown()
            print("[MotionWatch] Headless processing finished.")
    else:
        # Launch modern Surveillance Station GUI
        from ui import SurveillanceUI
        ui = SurveillanceUI(controller)
        controller.set_ui(ui)
        ui.mainloop()


if __name__ == "__main__":
    main()
