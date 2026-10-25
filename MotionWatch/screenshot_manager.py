"""
MotionWatch - Multi-Stage Incident Screenshot Manager
Saves rolling-buffer before frames, detection frame, and scheduled post-alert frames (after_1s, after_3s).
Generates incident JSON manifests and thumbnails.
"""

import os
import json
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from config import (
    INCIDENTS_DIR,
    BUFFER_SECONDS,
    BEFORE_CAPTURE_SECONDS,
    AFTER_1S_CAPTURE_SECONDS,
    AFTER_3S_CAPTURE_SECONDS
)


class IncidentCaptureTask:
    """Manages the lifecycle of an incident's scheduled screenshots."""

    def __init__(
        self,
        event_data: Dict[str, Any],
        incident_dir: Path,
        trigger_frame_idx: int,
        fps: float,
        video_source: str
    ):
        self.event_data = event_data
        self.incident_dir = incident_dir
        self.trigger_frame_idx = trigger_frame_idx
        self.fps = max(1.0, fps)
        self.video_source = video_source
        
        # Frame targets
        self.target_frame_1s = trigger_frame_idx + int(round(AFTER_1S_CAPTURE_SECONDS * self.fps))
        self.target_frame_3s = trigger_frame_idx + int(round(AFTER_3S_CAPTURE_SECONDS * self.fps))
        
        self.captured_files: Dict[str, str] = {}
        self.is_complete = False

    def check_and_capture(self, frame: np.ndarray, current_frame_idx: int) -> Tuple[bool, int]:
        """Saves scheduled future frames when reached. Returns (is_complete, count_saved)."""
        saved_count = 0
        if "after_1s.jpg" not in self.captured_files and current_frame_idx >= self.target_frame_1s:
            filename = "after_1s.jpg"
            save_path = self.incident_dir / filename
            cv2.imwrite(str(save_path), frame)
            self.captured_files[filename] = str(save_path)
            saved_count += 1
            self.write_manifest()

        if "after_3s.jpg" not in self.captured_files and current_frame_idx >= self.target_frame_3s:
            filename = "after_3s.jpg"
            save_path = self.incident_dir / filename
            cv2.imwrite(str(save_path), frame)
            self.captured_files[filename] = str(save_path)
            saved_count += 1
            self.is_complete = True
            self.write_manifest()
            return True, saved_count

        return False, saved_count

    def finalize(self):
        self.is_complete = True
        self.write_manifest()

    def write_manifest(self):
        """Writes or updates event.json manifest with currently captured screenshots."""
        manifest = {
            "event_id": self.event_data["event_id"],
            "event_type": self.event_data["event_type"],
            "timestamp": self.event_data["timestamp"],
            "video_source": str(self.video_source),
            "detection_method": self.event_data["detection_method"],
            "confidence": self.event_data.get("confidence", 0.0),
            "involved_track_ids": self.event_data.get("involved_track_ids", []),
            "details": self.event_data.get("details", ""),
            "screenshots": list(self.captured_files.keys()),
            "status": "COMPLETED" if self.is_complete else "IN_PROGRESS"
        }
        
        json_path = self.incident_dir / "event.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4)
        except Exception as e:
            print(f"[ScreenshotManager] Error saving incident manifest: {e}")


class ScreenshotManager:
    """Maintains circular frame buffer and schedules incident evidence captures."""

    def __init__(self, buffer_seconds: float = BUFFER_SECONDS):
        self.buffer_seconds = buffer_seconds
        # Buffer stores (frame_idx, timestamp, frame_copy)
        self.rolling_buffer: deque = deque()
        self.active_tasks: List[IncidentCaptureTask] = []
        self.total_saved_count = 0
        self.fps = 25.0
        self.lock = threading.Lock()

    def update_frame(self, frame: np.ndarray, frame_idx: int, fps: float):
        """Append latest frame to rolling buffer and advance pending capture tasks."""
        self.fps = max(1.0, fps)
        max_buffer_len = int(round(self.buffer_seconds * self.fps))
        
        with self.lock:
            # We copy frame to preserve state in buffer
            self.rolling_buffer.append((frame_idx, time.time(), frame.copy()))
            while len(self.rolling_buffer) > max_buffer_len:
                self.rolling_buffer.popleft()

            # Process pending incident captures
            remaining_tasks = []
            for task in self.active_tasks:
                finished, saved = task.check_and_capture(frame, frame_idx)
                self.total_saved_count += saved
                if not finished:
                    remaining_tasks.append(task)
            self.active_tasks = remaining_tasks

    def trigger_incident_capture(
        self,
        event_data: Dict[str, Any],
        current_frame: np.ndarray,
        trigger_frame_idx: int,
        fps: float,
        video_source: str = "Unknown"
    ) -> Path:
        """
        Creates incident directory and immediately saves:
        1. before.jpg (from rolling buffer approx 1.5s prior)
        2. detected.jpg (current trigger frame)
        3. Schedules after_1s.jpg and after_3s.jpg
        """
        self.fps = max(1.0, fps)
        event_id = event_data["event_id"]
        safe_source_name = Path(video_source).stem.replace(" ", "_") if video_source != "Unknown" else "cctv"
        incident_dirname = f"event_{time.strftime('%Y%m%d_%H%M%S')}_{safe_source_name}"
        incident_dir = INCIDENTS_DIR / incident_dirname
        
        # Ensure unique directory
        counter = 1
        base_dir = incident_dir
        while incident_dir.exists():
            incident_dir = Path(f"{base_dir}_{counter}")
            counter += 1

        incident_dir.mkdir(parents=True, exist_ok=True)
        event_data["incident_folder"] = str(incident_dir)

        task = IncidentCaptureTask(
            event_data=event_data,
            incident_dir=incident_dir,
            trigger_frame_idx=trigger_frame_idx,
            fps=self.fps,
            video_source=video_source
        )

        with self.lock:
            # 1. Save before.jpg from buffer
            target_before_frame = trigger_frame_idx - int(round(BEFORE_CAPTURE_SECONDS * self.fps))
            best_before_frame = None
            min_delta = float("inf")

            for f_idx, _, buf_frame in self.rolling_buffer:
                delta = abs(f_idx - target_before_frame)
                if delta < min_delta:
                    min_delta = delta
                    best_before_frame = buf_frame

            if best_before_frame is None and len(self.rolling_buffer) > 0:
                best_before_frame = self.rolling_buffer[0][2]
            elif best_before_frame is None:
                best_before_frame = current_frame

            before_path = incident_dir / "before.jpg"
            cv2.imwrite(str(before_path), best_before_frame)
            task.captured_files["before.jpg"] = str(before_path)
            self.total_saved_count += 1

            # 2. Save detected.jpg immediately
            detected_path = incident_dir / "detected.jpg"
            cv2.imwrite(str(detected_path), current_frame)
            task.captured_files["detected.jpg"] = str(detected_path)
            self.total_saved_count += 1

            # Save initial manifest in case video stops early
            task.finalize()
            task.is_complete = False  # Keep listening for 1s and 3s

            # Add to pending capture list
            self.active_tasks.append(task)

        print(f"[ScreenshotManager] Saved initial evidence to: {incident_dir}")
        return incident_dir

    def flush_remaining(self, last_frame: np.ndarray):
        """Forces completion of pending tasks if video terminates before after_3s occurs."""
        with self.lock:
            for task in self.active_tasks:
                if "after_1s.jpg" not in task.captured_files:
                    path = task.incident_dir / "after_1s.jpg"
                    cv2.imwrite(str(path), last_frame)
                    task.captured_files["after_1s.jpg"] = str(path)
                    self.total_saved_count += 1

                if "after_3s.jpg" not in task.captured_files:
                    path = task.incident_dir / "after_3s.jpg"
                    cv2.imwrite(str(path), last_frame)
                    task.captured_files["after_3s.jpg"] = str(path)
                    self.total_saved_count += 1

                task.finalize()
            self.active_tasks.clear()

    def get_screenshot_count(self) -> int:
        return self.total_saved_count
