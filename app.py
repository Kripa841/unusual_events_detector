"""
MotionWatch — AI-Based CCTV Theft Detection & Automated Alert System
Interactive Cloud & Recruiter Showcase Web Dashboard
Built with Streamlit, YOLOv8, ByteTrack, and Spatial-Temporal Kinematic Analytics.
"""

import os
import sys
import time
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure MotionWatch modules can be imported
ROOT_DIR = Path(__file__).resolve().parent
MOTIONWATCH_DIR = ROOT_DIR / "MotionWatch"
if str(MOTIONWATCH_DIR) not in sys.path:
    sys.path.insert(0, str(MOTIONWATCH_DIR))

from config import (
    MODEL_PATH,
    COLOR_BBOX_PERSON_BGR,
    COLOR_BBOX_ITEM_BGR,
    COLOR_BBOX_SUSPECT_BGR,
    DEFAULT_THEFT_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_CONFIDENCE
)
from detector import YOLODetector
from tracker import KinematicTracker
from theft_analyzer import TheftAnalyzer

# Page configuration
st.set_page_config(
    page_title="MotionWatch AI — Autonomous CCTV Theft Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern surveillance workstation aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0b0f17;
    }
    .main-header {
        background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
        padding: 18px 24px;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        margin-bottom: 20px;
    }
    .main-title {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .badge-alert {
        background-color: #ef4444;
        color: #ffffff;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        animation: pulse 1.5s infinite;
        text-align: center;
        font-size: 15px;
        margin-bottom: 12px;
    }
    .badge-normal {
        background-color: #10b981;
        color: #ffffff;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
        font-size: 15px;
        margin-bottom: 12px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    .stMetric {
        background: #1e293b;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


def get_audio_html(audio_path: Path) -> str:
    """Generates an invisible HTML5 autoplay audio tag for emergency siren playback."""
    if not audio_path.exists():
        return ""
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            return f"""
            <audio autoplay style="display:none;">
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
            """
    except Exception:
        return ""


@st.cache_resource
def load_detector():
    """Caches YOLOv8 detector instance across Streamlit reruns."""
    return YOLODetector(conf=DEFAULT_DETECTION_CONFIDENCE)


def get_curated_videos() -> Dict[str, Dict[str, Any]]:
    """Returns the curated CCTV incident case studies available in the project."""
    candidates = [
        (
            "Case Study 1: Purse & Phone Theft (mixkit-thief-purse.mp4)",
            ROOT_DIR / "mixkit-thief-silently-steals-a-womans-cell-phone-from-a-purse-31354-hd-ready (1).mp4",
            "Targeted handbag / phone extraction. Detects handbag, intimate proximity breach, and smooth item snatch.",
            "Person, Handbag, Backpack, Cell Phone"
        ),
        (
            "Case Study 2: Surveillance Robbery Interaction (Robbery_Suspects.mp4)",
            ROOT_DIR / "Robbery_Suspects_Caught_on_Surveillance_Camera.mp4",
            "CCTV feed capturing coordinated intercept, boundary contact, and rapid escape.",
            "Person, Vehicle, Bag, Suitcase"
        ),
        (
            "Case Study 3: Pedestrian Snatch & Sprint Escape (Mynewsdesk.mp4)",
            MOTIONWATCH_DIR / "Mynewsdesk.mp4",
            "Outdoor pedestrian corridor. Rapid convergence from behind, snatching bag/belongings, followed by sprint divergence.",
            "Person, Handbag, Phone, Backpack"
        ),
        (
            "Case Study 4: Sidewalk Snatch & Run Incident (sample.mp4)",
            ROOT_DIR / "sample (1).mp4" if (ROOT_DIR / "sample (1).mp4").exists() else MOTIONWATCH_DIR / "sample.mp4",
            "Street-level CCTV camera. Proximity interaction followed by high-speed acceleration escape.",
            "Person, Backpack, Handbag"
        ),
        (
            "Case Study 5: Rapid Divergence Theft Scenario (sample2.mp4)",
            MOTIONWATCH_DIR / "sample2.mp4",
            "Public walkway surveillance. Demonstrates multi-entity trajectory persistence and escape velocity ratio.",
            "Person, Handbag, Suitcase"
        ),
    ]

    videos = {}
    for title, path, desc, classes in candidates:
        if path.exists():
            videos[title] = {
                "path": path,
                "desc": desc,
                "target_classes": classes
            }
    return videos


def render_annotations(frame: np.ndarray, detections: list, active_tracks: dict, suspect_id: int = -1) -> np.ndarray:
    """Draws bounding boxes, tracking IDs, carried items, and trajectories."""
    out = frame.copy()

    for det in detections:
        tid = det["track_id"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cname = det["class_name"]

        if tid == suspect_id and suspect_id != -1:
            color = COLOR_BBOX_SUSPECT_BGR
            box_thickness = 3
            label = f"ALERT: SUSPECT ID {tid} ({cname})"
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

        # Label badge
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(out, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw trajectory history
        if tid in active_tracks:
            centers = active_tracks[tid].centers[-20:]
            for k in range(1, len(centers)):
                pt1 = (int(centers[k - 1][0]), int(centers[k - 1][1]))
                pt2 = (int(centers[k][0]), int(centers[k][1]))
                cv2.line(out, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

    return out


# ================= MAIN APPLICATION =================
def main():
    # Header Bar
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚨 MotionWatch AI — CCTV Theft & Snatch Detection System</h1>
        <p class="main-subtitle">Deep Learning Autonomous Surveillance • YOLOv8 + ByteTrack + Spatial-Temporal Kinematic Modeling</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Curated Video Showcase & Configuration
    with st.sidebar:
        st.subheader("🎯 HR & Recruiter Demonstration")
        st.info("To maintain high surveillance benchmarking standards, testing is restricted to curated CCTV incident case studies.")

        curated_videos = get_curated_videos()
        if not curated_videos:
            st.error("No sample videos found in MotionWatch directory.")
            return

        selected_video_name = st.selectbox(
            "Select CCTV Case Study Video:",
            options=list(curated_videos.keys()),
            index=0
        )
        selected_meta = curated_videos[selected_video_name]

        st.markdown(f"**Scenario Overview:** {selected_meta['desc']}")
        st.markdown(f"**Target Classes:** `{selected_meta['target_classes']}`")
        st.divider()

        st.subheader("⚙ Detection Parameters")
        theft_thresh = st.slider(
            "Theft Suspicion Threshold",
            min_value=0.20,
            max_value=0.80,
            value=float(DEFAULT_THEFT_CONFIDENCE_THRESHOLD),
            step=0.05,
            help="Sensitivity for triggering the emergency alert and siren."
        )

        det_thresh = st.slider(
            "YOLO Detection Sensitivity",
            min_value=0.15,
            max_value=0.60,
            value=float(DEFAULT_DETECTION_CONFIDENCE),
            step=0.05,
            help="Confidence threshold for bounding box detection."
        )

        enable_siren = st.checkbox("🔊 Enable Browser Audio Siren", value=True)
        frame_skip = st.select_slider(
            "Inference Speed Optimization",
            options=[1, 2, 3],
            value=1,
            format_func=lambda x: f"Every {x} frame(s) ({'Full Real-time' if x==1 else 'High FPS'})"
        )

    # Telemetry Row
    col_stat, col_tracks, col_items, col_fps = st.columns(4)
    with col_stat:
        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="badge-normal">🟢 SYSTEM STANDBY</div>', unsafe_allow_html=True)
    with col_tracks:
        tracks_metric = st.empty()
        tracks_metric.metric("Active Entities", "0 Persons / 0 Total")
    with col_items:
        items_metric = st.empty()
        items_metric.metric("Carried Belongings", "None Detected")
    with col_fps:
        fps_metric = st.empty()
        fps_metric.metric("Inference Rate", "0.0 FPS")

    # Alarm and Siren Banner Container
    alarm_banner_placeholder = st.empty()
    audio_placeholder = st.empty()

    # Split View: Left Video Canvas, Right Analytics & Evidence
    col_video, col_evidence = st.columns([1.6, 1.0])

    with col_video:
        st.subheader("📹 Live Surveillance Feed")
        video_canvas = st.empty()
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_btn = st.button("▶ Run Live Surveillance Analysis", type="primary", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹ Stop / Reset", use_container_width=True)

    with col_evidence:
        st.subheader("📸 Incident Snapshot Evidence")
        evidence_placeholder = st.empty()
        evidence_placeholder.info("Snapshot evidence bundle (Before, Detected, +1s, +3s) will appear here upon incident detection.")

    # Bottom Audit Log
    st.subheader("📋 Autonomous Incident Audit Trail")
    audit_table_placeholder = st.empty()

    if stop_btn:
        st.session_state["running"] = False
        st.rerun()

    if run_btn:
        st.session_state["running"] = True

        # Initialize engines
        detector = load_detector()
        detector.set_confidence(det_thresh)
        tracker = KinematicTracker()
        analyzer = TheftAnalyzer(confidence_threshold=theft_thresh)

        video_path = selected_meta["path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            st.error(f"Failed to open video source: {video_path}")
            return

        fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
        alarm_sound_path = MOTIONWATCH_DIR / "assets" / "alarm.wav"

        # Evidence capture state
        rolling_buffer: List[np.ndarray] = []
        max_buffer_frames = int(fps_val * 4.0)
        captured_events: List[Dict[str, Any]] = []
        active_suspect_id = -1
        alert_cooldown_counter = 0

        frame_idx = 0
        start_time = time.time()
        fps_calc = 0.0

        progress_bar = st.progress(0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        while cap.isOpened() and st.session_state.get("running", True):
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress_bar.progress(min(1.0, frame_idx / total_frames))

            # Maintain rolling ring-buffer for before-incident snapshot
            rolling_buffer.append(frame.copy())
            if len(rolling_buffer) > max_buffer_frames:
                rolling_buffer.pop(0)

            # Skip frames if user selected optimization
            if frame_idx % frame_skip != 0:
                continue

            # Aspect-safe resize for inference efficiency
            h, w = frame.shape[:2]
            scale = 960.0 / max(w, 1)
            frame_resized = cv2.resize(frame, (960, int(h * scale)))

            t0 = time.time()
            # YOLO detection with category-aware thresholds
            detections = detector.detect_and_track(frame_resized, persist=True)

            # Kinematic Tracking & Carried Item Persistence
            active_tracks = tracker.update(detections, frame_idx, fps_val)

            # Theft & Snatch Interaction Recognition
            event = analyzer.analyze_frame(frame_resized, active_tracks, frame_idx, fps_val)

            t1 = time.time()
            fps_calc = 1.0 / max(0.001, t1 - t0)

            # Check if incident detected
            if event is not None:
                active_suspect_id = event.get("suspect_id", -1)
                alert_cooldown_counter = int(fps_val * 3.5)
                captured_events.append(event)

                # Visual Warning Banner
                alarm_banner_placeholder.markdown(f"""
                <div class="badge-alert">
                    🚨 ⚠ SUSPECTED THEFT DETECTED: {event['event_id']} (Confidence: {int(event['confidence']*100)}%)<br>
                    <span style="font-size:12px; font-weight:normal;">{event['details']}</span>
                </div>
                """, unsafe_allow_html=True)
                status_placeholder.markdown('<div class="badge-alert">🚨 ALERT ACTIVE</div>', unsafe_allow_html=True)

                # Sound Siren
                if enable_siren:
                    audio_html = get_audio_html(alarm_sound_path)
                    if audio_html:
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

                # Capture Evidence Snapshots
                before_idx = max(0, len(rolling_buffer) - int(fps_val * 1.5))
                before_img = cv2.cvtColor(rolling_buffer[before_idx], cv2.COLOR_BGR2RGB)
                detected_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with col_evidence:
                    ev_col1, ev_col2 = st.columns(2)
                    with ev_col1:
                        st.image(before_img, caption="1. Before Incident (Buffer)", use_container_width=True)
                    with ev_col2:
                        st.image(detected_img, caption="2. Incident Trigger", use_container_width=True)

            elif alert_cooldown_counter > 0:
                alert_cooldown_counter -= 1
            else:
                active_suspect_id = -1
                alarm_banner_placeholder.empty()
                status_placeholder.markdown('<div class="badge-normal">🟢 SURVEILLANCE ACTIVE</div>', unsafe_allow_html=True)

            # Render annotations on frame
            annotated_frame = render_annotations(frame_resized, detections, active_tracks, active_suspect_id)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_canvas.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Update Metrics
            person_count = len([t for t in active_tracks.values() if t.is_person and t.missed_frames == 0])
            all_carried: List[str] = []
            for t in active_tracks.values():
                if t.is_person and t.carried_items:
                    all_carried.extend(t.carried_items)

            tracks_metric.metric("Active Entities", f"{person_count} Persons / {len(active_tracks)} Total")
            items_metric.metric("Carried Belongings", ", ".join(set(all_carried)) if all_carried else "None")
            fps_metric.metric("Inference Rate", f"{fps_calc:.1f} FPS")

            # Update Event Log Table
            if captured_events:
                df = pd.DataFrame(captured_events)[["event_id", "timestamp", "confidence", "suspect_id", "victim_id", "details"]]
                audit_table_placeholder.dataframe(df, use_container_width=True)

        cap.release()
        status_placeholder.markdown('<div class="badge-normal">✅ ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
        st.session_state["running"] = False


if __name__ == "__main__":
    main()

