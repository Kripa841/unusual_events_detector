"""
MotionWatch - Event Logging & Evidence Record Manager
Maintains persistent CSV log files, JSON event records, and asynchronous video recording.
"""

import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np

from config import (
    LOG_CSV_PATH,
    CSV_COLUMNS,
    ANNOTATED_DIR
)


class EventLogger:
    """Thread-safe event logger for CSV audits and video evidence recording."""

    def __init__(self, csv_path: Path = LOG_CSV_PATH):
        self.csv_path = csv_path
        self._ensure_csv_exists()
        self.events_cache: List[Dict[str, Any]] = []
        self._load_cached_events()

    def _ensure_csv_exists(self):
        """Creates the CSV file with headers if it does not exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def _load_cached_events(self):
        """Loads previous events from CSV for GUI inspection."""
        if not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.events_cache.append(dict(row))
        except Exception as e:
            print(f"[Logger] Error reading CSV history: {e}")

    def log_event(self, event_data: Dict[str, Any], video_source: str = "Unknown") -> bool:
        """Appends a detected incident to the CSV log."""
        self._ensure_csv_exists()
        
        row = [
            event_data.get("event_id", "N/A"),
            event_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            str(Path(video_source).name),
            event_data.get("event_type", "SUSPECTED_THEFT"),
            event_data.get("detection_method", "N/A"),
            f"{event_data.get('confidence', 0.0):.3f}",
            str(event_data.get("involved_track_ids", [])),
            str(event_data.get("incident_folder", "N/A")),
            str(event_data.get("acknowledged", False))
        ]

        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Update memory cache
            row_dict = dict(zip(CSV_COLUMNS, row))
            self.events_cache.append(row_dict)
            print(f"[Logger] Logged incident to CSV: {event_data.get('event_id')}")
            return True
        except Exception as e:
            print(f"[Logger] Failed to write event to CSV: {e}")
            return False

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent events in reverse chronological order."""
        return list(reversed(self.events_cache[-limit:]))


class AsyncVideoWriter:
    """Non-blocking video recorder for saving annotated CCTV output."""

    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        self.is_open = self.writer.isOpened()
        if not self.is_open:
            print(f"[AsyncVideoWriter] Warning: Failed to open video writer for {output_path}")

    def write_frame(self, frame: np.ndarray):
        """Writes annotated frame if writer is active."""
        if self.is_open and self.writer is not None:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def release(self):
        """Safely closes video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.is_open = False
            print(f"[AsyncVideoWriter] Saved annotated video to: {self.output_path}")
