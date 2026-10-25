"""
MotionWatch - Suspected Theft Event & Temporal Interaction Analyzer
Evaluates multi-frame spatial-temporal kinematics, proximity interactions,
motion energy disturbances, and item transfers to detect suspected snatch/theft.
"""

import time
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

from tracker import TrackRecord
from config import (
    DEFAULT_THEFT_CONFIDENCE_THRESHOLD,
    PROXIMITY_INTERACTION_DISTANCE_PX,
    SNATCH_ESCAPE_VELOCITY_RATIO,
    MIN_ESCAPE_SPEED_PX,
    MAX_INTERACTION_DURATION_FRAMES,
    MIN_APPROACH_FRAMES,
    MOTION_ENERGY_STRUGGLE_THRESHOLD,
    ALERT_COOLDOWN_SECONDS
)


class PairInteractionState:
    """Tracks the multi-frame interaction progression between a pair of tracks."""

    def __init__(self, id_a: int, id_b: int, start_frame: int):
        self.pair_key = (min(id_a, id_b), max(id_a, id_b))
        self.id_a = id_a
        self.id_b = id_b
        self.start_frame = start_frame
        self.last_frame = start_frame
        
        self.distances: List[float] = []
        self.frame_indices: List[int] = []
        self.approach_detected = False
        self.min_distance = float("inf")
        self.min_dist_frame = start_frame
        self.motion_energies: List[float] = []
        self.has_triggered_alert = False


class TheftAnalyzer:
    """
    Analyzes temporal interactions between tracked entities to identify
    anomalous theft patterns such as rapid intercept & snatch, drive-by snatch,
    and loitering followed by abrupt bolting.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_THEFT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.interactions: Dict[Tuple[int, int], PairInteractionState] = {}
        self.prev_gray: Optional[np.ndarray] = None
        self.last_alert_time: float = 0.0
        self.event_counter = 0

    def analyze_frame(
        self,
        frame: np.ndarray,
        tracks: Dict[int, TrackRecord],
        frame_idx: int,
        fps: float = 25.0
    ) -> Optional[Dict[str, Any]]:
        """
        Processes current frame and active tracks.
        Returns an incident event dictionary if suspected theft is detected, else None.
        """
        # Convert to grayscale for motion energy difference calculation
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate full-frame motion diff if previous frame exists
        motion_diff = None
        if self.prev_gray is not None and self.prev_gray.shape == curr_gray.shape:
            motion_diff = cv2.absdiff(curr_gray, self.prev_gray)
            
        self.prev_gray = curr_gray

        # Extract active person tracks and vehicle tracks (motorcycles/bikes often used in snatching)
        persons = [t for t in tracks.values() if t.is_person and t.missed_frames == 0]
        vehicles = [t for t in tracks.values() if not t.is_person and t.missed_frames == 0 and t.class_id in [1, 2, 3]]
        
        # All potential active agents
        active_agents = persons + vehicles
        if len(active_agents) < 2:
            self._cleanup_old_interactions(frame_idx)
            return None

        # Check all distinct pairs
        now = time.time()
        for i in range(len(active_agents)):
            for j in range(i + 1, len(active_agents)):
                agent_a = active_agents[i]
                agent_b = active_agents[j]
                
                # At least one agent should be a person
                if not (agent_a.is_person or agent_b.is_person):
                    continue

                event = self._evaluate_pair(agent_a, agent_b, frame_idx, motion_diff, fps, now)
                if event is not None:
                    self._cleanup_old_interactions(frame_idx)
                    return event

        self._cleanup_old_interactions(frame_idx)
        return None

    def _evaluate_pair(
        self,
        t_a: TrackRecord,
        t_b: TrackRecord,
        frame_idx: int,
        motion_diff: Optional[np.ndarray],
        fps: float,
        now: float
    ) -> Optional[Dict[str, Any]]:
        """Evaluates interaction history between two track records for snatch/theft signatures."""
        key = (min(t_a.track_id, t_b.track_id), max(t_a.track_id, t_b.track_id))
        
        pos_a = t_a.current_center
        pos_b = t_b.current_center
        dist = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])

        # Get or create interaction state
        if key not in self.interactions:
            self.interactions[key] = PairInteractionState(t_a.track_id, t_b.track_id, frame_idx)

        state = self.interactions[key]
        state.last_frame = frame_idx
        state.distances.append(dist)
        state.frame_indices.append(frame_idx)

        # Keep history to max interaction window
        if len(state.distances) > MAX_INTERACTION_DURATION_FRAMES:
            state.distances = state.distances[-MAX_INTERACTION_DURATION_FRAMES:]
            state.frame_indices = state.frame_indices[-MAX_INTERACTION_DURATION_FRAMES:]

        if dist < state.min_distance:
            state.min_distance = dist
            state.min_dist_frame = frame_idx

        # Measure motion energy in inter-agent space if motion diff is available
        overlap_energy = 0.0
        if motion_diff is not None:
            overlap_energy = self._compute_overlap_motion_energy(t_a.current_bbox, t_b.current_bbox, motion_diff)
            state.motion_energies.append(overlap_energy)
            if len(state.motion_energies) > MAX_INTERACTION_DURATION_FRAMES:
                state.motion_energies = state.motion_energies[-MAX_INTERACTION_DURATION_FRAMES:]

        # Need at least MIN_APPROACH_FRAMES of observation
        if len(state.distances) < MIN_APPROACH_FRAMES:
            return None

        # Check if already triggered recently
        if state.has_triggered_alert:
            return None

        # Cooldown check
        if (now - self.last_alert_time) < ALERT_COOLDOWN_SECONDS:
            return None

        # ----------------- THEFT DETECTION KINEMATICS -----------------
        # Pattern 1: Rapid Convergence -> Close Proximity -> Rapid Divergent Escape / Sprint
        initial_dist = state.distances[0]
        min_dist = state.min_distance
        current_dist = state.distances[-1]
        
        # Did the tracks converge or make close proximity contact?
        has_converged = (initial_dist - min_dist > 30.0) or (min_dist <= PROXIMITY_INTERACTION_DISTANCE_PX)
        
        # Divergence or high speed escape movement after close encounter
        is_diverging = (current_dist > min_dist + 15.0) and (frame_idx > state.min_dist_frame + 1)
        is_sprint_escape = (t_a.current_speed >= MIN_ESCAPE_SPEED_PX or t_b.current_speed >= MIN_ESCAPE_SPEED_PX)

        if (has_converged and is_diverging) or (min_dist <= PROXIMITY_INTERACTION_DISTANCE_PX and is_sprint_escape):
            # Determine which track is moving faster (the suspect)
            speed_a = t_a.current_speed
            speed_b = t_b.current_speed
            
            if speed_a >= speed_b:
                suspect, victim = t_a, t_b
                suspect_speed, victim_speed = speed_a, speed_b
            else:
                suspect, victim = t_b, t_a
                suspect_speed, victim_speed = speed_b, speed_a

            # Compute approach speed from the approach phase
            approach_frames = max(1, state.min_dist_frame - state.start_frame)
            approach_dist_delta = max(0.0, initial_dist - min_dist)
            approach_rate = approach_dist_delta / approach_frames

            # Compute Escape Velocity Ratio
            baseline_speed = max(approach_rate, victim_speed, 1.5)
            escape_ratio = suspect_speed / baseline_speed

            # Motion energy spike during interaction
            avg_energy = float(np.mean(state.motion_energies)) if state.motion_energies else 0.0

            # Context bonus for carried personal items (phone, handbag, backpack)
            item_bonus = 0.15 if (len(victim.carried_items) > 0 or len(suspect.carried_items) > 0) else 0.0

            # ----------------- COMPOSITE CONFIDENCE CALCULATION -----------------
            # 1. Proximity score (0.05 to 0.35)
            proximity_score = max(0.05, min(0.35, (PROXIMITY_INTERACTION_DISTANCE_PX - min_dist) / PROXIMITY_INTERACTION_DISTANCE_PX * 0.35))
            
            # 2. Escape Speed score (0.05 to 0.40)
            speed_score = min(0.40, max(0.05, (suspect_speed / 15.0) * 0.40))

            # 3. Escape Velocity Ratio score (0.0 to 0.20)
            escape_ratio_score = 0.0
            if escape_ratio >= SNATCH_ESCAPE_VELOCITY_RATIO:
                escape_ratio_score = min(0.20, max(0.05, (escape_ratio - 1.0) / 2.0 * 0.20))

            # 4. Motion energy struggle disturbance (0.0 to 0.20)
            energy_score = min(0.20, max(0.05, (avg_energy / MOTION_ENERGY_STRUGGLE_THRESHOLD) * 0.20))

            # 5. Approach convergence bonus
            conv_bonus = 0.15 if (initial_dist - min_dist >= 30.0) else 0.0

            # Total calculated confidence score
            calculated_confidence = proximity_score + speed_score + escape_ratio_score + energy_score + conv_bonus + item_bonus
            calculated_confidence = min(0.99, max(0.10, calculated_confidence))

            # Verify if threshold is met and escape velocity / sprint criteria is satisfied
            if calculated_confidence >= self.confidence_threshold and (suspect_speed >= MIN_ESCAPE_SPEED_PX or escape_ratio >= SNATCH_ESCAPE_VELOCITY_RATIO):
                state.has_triggered_alert = True
                self.last_alert_time = now
                self.event_counter += 1

                event_id = f"THEFT_{time.strftime('%Y%m%d_%H%M%S')}_{self.event_counter:02d}"
                event_type = "SUSPECTED_SNATCH_THEFT"
                
                details = (
                    f"Close interaction (dist {int(min_dist)}px) followed by "
                    f"rapid departure by ID {suspect.track_id} (speed: {suspect_speed:.1f}px/f, ratio: {escape_ratio:.2f}x)."
                )

                return {
                    "event_id": event_id,
                    "event_type": event_type,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_unix": now,
                    "detection_method": "Temporal Kinematic + Motion Energy Interaction Analysis",
                    "confidence": round(float(calculated_confidence), 3),
                    "suspect_id": suspect.track_id,
                    "victim_id": victim.track_id,
                    "involved_track_ids": [suspect.track_id, victim.track_id],
                    "details": details,
                    "is_manual_test": False
                }

        return None

    def _compute_overlap_motion_energy(self, bbox_a: List[float], bbox_b: List[float], motion_diff: np.ndarray) -> float:
        """Calculates pixel motion variance in the union / interaction region between two bboxes."""
        h, w = motion_diff.shape
        x1 = max(0, int(min(bbox_a[0], bbox_b[0])))
        y1 = max(0, int(min(bbox_a[1], bbox_b[1])))
        x2 = min(w, int(max(bbox_a[2], bbox_b[2])))
        y2 = min(h, int(max(bbox_a[3], bbox_b[3])))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        region = motion_diff[y1:y2, x1:x2]
        return float(np.mean(region))

    def trigger_manual_test(self, involved_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generates a clearly labeled manual test alert for verification of
        audio sirens, red warning overlay, screenshot capture, and CSV logging.
        """
        now = time.time()
        self.last_alert_time = now
        self.event_counter += 1

        event_id = f"TEST_{time.strftime('%Y%m%d_%H%M%S')}_{self.event_counter:02d}"
        return {
            "event_id": event_id,
            "event_type": "MANUAL_TEST_ALERT",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_unix": now,
            "detection_method": "User Manual Verification Test",
            "confidence": 1.00,
            "suspect_id": (involved_ids[0] if involved_ids else -1),
            "victim_id": (involved_ids[1] if involved_ids and len(involved_ids) > 1 else -1),
            "involved_track_ids": involved_ids or [],
            "details": "Manual test alert triggered via user interface to verify alarm and evidence capture.",
            "is_manual_test": True
        }

    def _cleanup_old_interactions(self, current_frame: int):
        """Prunes interaction states that have been inactive for too long."""
        stale_keys = [
            k for k, state in self.interactions.items()
            if (current_frame - state.last_frame) > MAX_INTERACTION_DURATION_FRAMES * 2
        ]
        for k in stale_keys:
            del self.interactions[k]

    def set_threshold(self, threshold: float):
        """Update detection confidence sensitivity threshold."""
        self.confidence_threshold = max(0.20, min(0.95, threshold))

    def reset(self):
        """Reset internal state."""
        self.interactions.clear()
        self.prev_gray = None
