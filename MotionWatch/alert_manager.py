"""
MotionWatch - Alert Manager Module
Handles non-blocking audible alarms, red warning visual banners,
pulsing alert borders, and alarm acknowledgement.
"""

import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

from config import (
    ALARM_SOUND_PATH,
    AUDIO_ALERT_DURATION_SECONDS,
    VISUAL_ALERT_DURATION_SECONDS,
    COLOR_ALERT_BGR,
    COLOR_NORMAL_BGR,
    COLOR_WARNING_BGR
)


class AlertManager:
    """Coordinates audible sirens, visual warning states, and user acknowledgement."""

    def __init__(self):
        self.active_event: Optional[Dict[str, Any]] = None
        self.alert_start_time: float = 0.0
        self.is_alert_active: bool = False
        self.audio_playing: bool = False
        self.is_acknowledged: bool = False
        
        self.audio_thread: Optional[threading.Thread] = None
        self._stop_audio_event = threading.Event()
        self.sound_file = str(ALARM_SOUND_PATH.resolve()) if ALARM_SOUND_PATH.exists() else None

    def trigger_alert(self, event_data: Dict[str, Any]):
        """Triggers the full audio-visual alarm state."""
        self.active_event = event_data
        self.alert_start_time = time.time()
        self.is_alert_active = True
        self.is_acknowledged = False
        
        print(f"\n[ALERT] >>> {event_data.get('event_type')} DETECTED! Event ID: {event_data.get('event_id')} <<<")
        self.start_audio_alarm()

    def start_audio_alarm(self):
        """Starts audible alarm in a non-blocking manner."""
        if self.audio_playing:
            return  # Prevent overlapping duplicate sirens

        self.audio_playing = True
        self._stop_audio_event.clear()

        # On Windows, winsound with SND_ASYNC | SND_LOOP is completely non-blocking and native
        try:
            import winsound
            if self.sound_file and Path(self.sound_file).exists():
                winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            else:
                # Threaded beep fallback
                self.audio_thread = threading.Thread(target=self._threaded_beep_loop, daemon=True)
                self.audio_thread.start()
        except Exception as e:
            # Fallback for non-windows or audio errors
            self.audio_thread = threading.Thread(target=self._threaded_beep_loop, daemon=True)
            self.audio_thread.start()

    def _threaded_beep_loop(self):
        """Fallback beep loop if sound file is unavailable."""
        try:
            import winsound
            while not self._stop_audio_event.is_set():
                winsound.Beep(1400, 200)
                if self._stop_audio_event.is_set():
                    break
                winsound.Beep(900, 200)
        except Exception:
            # Terminal bell fallback
            while not self._stop_audio_event.is_set():
                print("\a", end="", flush=True)
                time.sleep(0.4)

    def stop_audio_alarm(self):
        """Stops the audible siren immediately."""
        if not self.audio_playing:
            return

        self.audio_playing = False
        self._stop_audio_event.set()

        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def acknowledge_alarm(self):
        """Acknowledge current alarm by user."""
        self.is_acknowledged = True
        self.stop_audio_alarm()
        self.is_alert_active = False
        if self.active_event:
            self.active_event["acknowledged"] = True
        print("[AlertManager] Alarm acknowledged by operator.")

    def update(self):
        """Periodic status update to automatically expire alert duration."""
        if not self.is_alert_active and not self.audio_playing:
            return

        elapsed = time.time() - self.alert_start_time

        # Expire audio alarm after duration
        if self.audio_playing and elapsed >= AUDIO_ALERT_DURATION_SECONDS:
            self.stop_audio_alarm()

        # Expire visual warning after duration
        if self.is_alert_active and elapsed >= VISUAL_ALERT_DURATION_SECONDS:
            self.is_alert_active = False

    def render_overlay(self, frame: np.ndarray, is_monitoring: bool, fps_display: float) -> np.ndarray:
        """
        Renders the security surveillance status overlay on top of the OpenCV frame.
        - Red pulsing border and warning banner during SUSPECTED THEFT.
        - Green border and neutral status during normal monitoring.
        """
        h, w = frame.shape[:2]
        out_frame = frame.copy()

        if self.is_alert_active and self.active_event:
            # Alert mode: Bright red border and large warning banner
            border_thickness = 12
            cv2.rectangle(out_frame, (0, 0), (w, h), COLOR_ALERT_BGR, border_thickness)

            # Top warning banner
            banner_h = 75
            overlay = out_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 200), -1)
            cv2.addWeighted(overlay, 0.85, out_frame, 0.15, 0, out_frame)

            # Warning text
            event_type = self.active_event.get("event_type", "SUSPECTED THEFT")
            if self.active_event.get("is_manual_test", False):
                title_text = "[TEST] ⚠ MANUAL TEST ALERT TRIGGERED"
            else:
                title_text = "⚠ SUSPECTED THEFT DETECTED"

            cv2.putText(out_frame, title_text, (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            # Secondary info bar
            eid = self.active_event.get("event_id", "N/A")
            conf = self.active_event.get("confidence", 0.0)
            tstamp = self.active_event.get("timestamp", "")
            sub_text = f"ID: {eid} | Conf: {conf*100:.1f}% | Time: {tstamp}"
            cv2.putText(out_frame, sub_text, (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 255), 1, cv2.LINE_AA)

            # Bottom acknowledge hint
            cv2.rectangle(out_frame, (w - 280, h - 35), (w - 15, h - 10), (0, 0, 0), -1)
            cv2.putText(out_frame, "PRESS [A] TO ACKNOWLEDGE", (w - 270, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        else:
            # Normal surveillance mode: Subtle green status
            border_thickness = 3
            status_color = COLOR_NORMAL_BGR if is_monitoring else (120, 120, 120)
            cv2.rectangle(out_frame, (0, 0), (w, h), status_color, border_thickness)

            # Top mini status pill
            pill_w = 230
            cv2.rectangle(out_frame, (10, 10), (10 + pill_w, 42), (20, 24, 30), -1)
            cv2.rectangle(out_frame, (10, 10), (10 + pill_w, 42), (50, 60, 75), 1)

            # Pulsing green dot if active
            status_dot_color = (0, 255, 0) if is_monitoring else (100, 100, 100)
            cv2.circle(out_frame, (25, 26), 6, status_dot_color, -1)

            status_str = "MONITORING ACTIVE" if is_monitoring else "STANDBY / PAUSED"
            cv2.putText(out_frame, status_str, (38, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

            # FPS counter
            cv2.putText(out_frame, f"FPS: {fps_display:.1f}", (w - 95, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return out_frame
