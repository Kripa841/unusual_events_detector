"""
MotionWatch - Configuration Module
Centralized parameters for detection, tracking, theft analysis, alerts, and logging.
"""

from pathlib import Path

# Directories
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
INCIDENTS_DIR = OUTPUT_DIR / "incidents"
LOGS_DIR = OUTPUT_DIR / "logs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_videos"
ASSETS_DIR = PROJECT_DIR / "assets"
MODELS_DIR = PROJECT_DIR / "models"

# Ensure output directories exist
for directory in [OUTPUT_DIR, INCIDENTS_DIR, LOGS_DIR, ANNOTATED_DIR, ASSETS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Settings
# Default model: local yolov8n.pt if present, else fallback to standard download
LOCAL_YOLO_PATH = PROJECT_DIR / "yolov8n.pt"
MODEL_PATH = str(LOCAL_YOLO_PATH if LOCAL_YOLO_PATH.exists() else "yolov8n.pt")

# COCO Classes of interest:
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 24: backpack, 26: handbag, 28: suitcase, 67: cell phone
DETECT_CLASSES = [0, 1, 2, 3, 24, 26, 28, 67]
PERSON_CLASS_ID = 0
ITEM_CLASS_IDS = [24, 26, 28, 67]
VEHICLE_CLASS_IDS = [1, 2, 3]

# Category-Specific Confidence Thresholds (avoids dropping small bags/suitcases)
DEFAULT_DETECTION_CONFIDENCE = 0.20
CONF_PERSON = 0.22
CONF_ITEM = 0.15       # Backpack (24), Handbag (26), Suitcase (28), Cell Phone (67)
CONF_VEHICLE = 0.20
RESIZE_WIDTH = 960  # Frame resizing for real-time inference efficiency

# Theft Kinematics & Temporal Analysis Thresholds
DEFAULT_THEFT_CONFIDENCE_THRESHOLD = 0.40  # Balanced sensitivity for real-world CCTV
PROXIMITY_INTERACTION_DISTANCE_PX = 180.0  # Max distance for close interaction in 960px frame
SNATCH_ESCAPE_VELOCITY_RATIO = 1.35        # Ratio of escape speed to baseline speed
MIN_ESCAPE_SPEED_PX = 6.5                  # Speed threshold (pixels/frame) to qualify as running/escape
MAX_INTERACTION_DURATION_FRAMES = 60       # Window within which snatch occurs (approx 2s)
MIN_APPROACH_FRAMES = 3                    # Minimum frames of observed approach
MOTION_ENERGY_STRUGGLE_THRESHOLD = 16.0    # Frame-differencing intensity inside overlap zone

# Alert Behavior
ALERT_COOLDOWN_SECONDS = 4.0               # Minimum seconds between triggering new alerts
VISUAL_ALERT_DURATION_SECONDS = 4.0        # Red warning interface duration
AUDIO_ALERT_DURATION_SECONDS = 5.0         # Siren duration unless acknowledged early
ALARM_SOUND_PATH = ASSETS_DIR / "alarm.wav"

# Screenshot Capture Timings (in seconds relative to event trigger)
BUFFER_SECONDS = 4.0                       # Rolling frame buffer length
BEFORE_CAPTURE_SECONDS = 1.5               # Capture frame ~1.5s before alert
AFTER_1S_CAPTURE_SECONDS = 1.0             # Capture frame 1.0s after alert
AFTER_3S_CAPTURE_SECONDS = 3.0             # Capture frame 3.0s after alert

# Logging
LOG_CSV_PATH = LOGS_DIR / "theft_events.csv"
CSV_COLUMNS = [
    "Event_ID",
    "Timestamp",
    "Video_Source",
    "Event_Type",
    "Detection_Method",
    "Confidence",
    "Involved_Track_IDs",
    "Incident_Folder",
    "Acknowledged"
]

# Visual Palette (BGR for OpenCV, Hex for Tkinter)
COLOR_NORMAL_BGR = (40, 200, 40)
COLOR_ALERT_BGR = (30, 30, 240)
COLOR_WARNING_BGR = (0, 165, 255)
COLOR_BBOX_PERSON_BGR = (255, 178, 50)
COLOR_BBOX_ITEM_BGR = (200, 50, 200)
COLOR_BBOX_SUSPECT_BGR = (0, 0, 255)

UI_THEME = {
    "bg_dark": "#0f1218",
    "panel_bg": "#181d26",
    "panel_border": "#283142",
    "accent_blue": "#2563eb",
    "accent_green": "#10b981",
    "accent_red": "#ef4444",
    "accent_orange": "#f59e0b",
    "text_main": "#f3f4f6",
    "text_muted": "#9ca3af",
}
