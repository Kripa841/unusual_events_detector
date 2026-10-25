"""
MotionWatch - Multi-Object Kinematic Tracker & History Manager
Maintains spatial-temporal track histories, velocities, accelerations, and carried items.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class TrackRecord:
    """Stores spatial-temporal kinematic history for a single tracked identity."""

    def __init__(self, track_id: int, class_id: int, class_name: str, initial_bbox: List[float], frame_idx: int):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.bboxes: List[List[float]] = [initial_bbox]
        
        cx = (initial_bbox[0] + initial_bbox[2]) / 2.0
        cy = (initial_bbox[1] + initial_bbox[3]) / 2.0
        self.centers: List[Tuple[float, float]] = [(cx, cy)]
        self.frame_indices: List[int] = [frame_idx]
        
        self.speeds: List[float] = [0.0]
        self.velocities: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.accelerations: List[float] = [0.0]
        
        self.missed_frames = 0
        self.is_person = (class_id == 0)
        self.carried_items: List[str] = []

    def update(self, bbox: List[float], frame_idx: int, fps: float = 25.0):
        """Update track with new bounding box and compute kinematic derivatives."""
        self.missed_frames = 0
        self.bboxes.append(bbox)
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.centers.append((cx, cy))
        self.frame_indices.append(frame_idx)
        
        # Keep maximum 90 frames of history (~3 seconds at 30fps)
        if len(self.centers) > 90:
            self.bboxes = self.bboxes[-90:]
            self.centers = self.centers[-90:]
            self.frame_indices = self.frame_indices[-90:]
            self.speeds = self.speeds[-90:]
            self.velocities = self.velocities[-90:]
            self.accelerations = self.accelerations[-90:]

        # Calculate velocity and speed
        if len(self.centers) >= 2:
            dt_frames = max(1, self.frame_indices[-1] - self.frame_indices[-2])
            prev_cx, prev_cy = self.centers[-2]
            vx = (cx - prev_cx) / dt_frames
            vy = (cy - prev_cy) / dt_frames
            speed = math.hypot(vx, vy)
            
            self.velocities.append((vx, vy))
            self.speeds.append(speed)
            
            # Acceleration
            if len(self.speeds) >= 2:
                accel = (self.speeds[-1] - self.speeds[-2]) / dt_frames
                self.accelerations.append(accel)
        else:
            self.velocities.append((0.0, 0.0))
            self.speeds.append(0.0)
            self.accelerations.append(0.0)

    @property
    def current_speed(self) -> float:
        """Moving average speed over the last 3-5 frames."""
        if not self.speeds:
            return 0.0
        window = self.speeds[-min(5, len(self.speeds)):]
        return float(np.mean(window))

    @property
    def current_acceleration(self) -> float:
        """Instantaneous acceleration."""
        return self.accelerations[-1] if self.accelerations else 0.0

    @property
    def current_center(self) -> Tuple[float, float]:
        return self.centers[-1]

    @property
    def current_bbox(self) -> List[float]:
        return self.bboxes[-1]


class KinematicTracker:
    """Coordinates active track records and performs object association."""

    def __init__(self, max_missed_frames: int = 20):
        self.tracks: Dict[int, TrackRecord] = {}
        self.max_missed_frames = max_missed_frames

    def update(self, detections: List[Dict[str, Any]], frame_idx: int, fps: float = 25.0) -> Dict[int, TrackRecord]:
        """Update all tracks with current frame detections."""
        seen_track_ids = set()

        for det in detections:
            tid = det["track_id"]
            if tid < 0:
                continue

            seen_track_ids.add(tid)
            if tid in self.tracks:
                self.tracks[tid].update(det["bbox"], frame_idx, fps)
            else:
                self.tracks[tid] = TrackRecord(
                    track_id=tid,
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    initial_bbox=det["bbox"],
                    frame_idx=frame_idx
                )

        # Increment missed frames for tracks that weren't detected
        stale_ids = []
        for tid, track in self.tracks.items():
            if tid not in seen_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    stale_ids.append(tid)

        # Remove dead tracks
        for tid in stale_ids:
            del self.tracks[tid]

        # Associate carried items (backpack, handbag, phone) with nearest overlapping person
        self._associate_items_with_persons(detections)

        return self.tracks

    def _associate_items_with_persons(self, detections: List[Dict[str, Any]]):
        """Checks if items are inside or very close to a person's bounding box."""
        persons = [d for d in detections if d["is_person"] and d["track_id"] in self.tracks]
        items = [d for d in detections if d["is_item"]]

        for p in persons:
            p_track = self.tracks[p["track_id"]]
            p_track.carried_items.clear()
            px1, py1, px2, py2 = p["bbox"]

            for it in items:
                ix1, iy1, ix2, iy2 = it["bbox"]
                icx, icy = it["center"]
                # If item center is inside person's expanded box
                if (px1 - 20 <= icx <= px2 + 20) and (py1 - 20 <= icy <= py2 + 20):
                    p_track.carried_items.append(it["class_name"])

    def get_person_tracks(self) -> List[TrackRecord]:
        """Returns list of active person tracks."""
        return [t for t in self.tracks.values() if t.is_person and t.missed_frames == 0]

    def reset(self):
        """Clear all tracking history."""
        self.tracks.clear()
