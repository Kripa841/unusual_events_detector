# MotionWatch — AI-Based CCTV Theft Detection & Automated Alert System

**MotionWatch** is an enterprise-grade Image Processing and Computer Vision (IPCV) surveillance analysis system designed to autonomously monitor CCTV video streams, identify **suspected theft and snatch-and-run incidents**, trigger immediate audio-visual alarms, and preserve multi-stage photographic evidence bundles.

Developed for high-reliability surveillance workstations, MotionWatch pairs deep-learning object detection with spatial-temporal kinematic interaction modeling to distinguish genuine suspicious theft signatures from ordinary ambient motion.

---

## 1. Project Objective

Ordinary surveillance systems rely on basic motion detection or zone intrusion, which trigger false alarms every time a person walks across the camera view. MotionWatch focuses on **interaction semantics and kinematics**:
1. **Multi-Entity Tracking**: Continuously tracks pedestrians, carried belongings (handbags, backpacks, phones), and getaway vehicles (motorcycles, bicycles).
2. **Theft & Snatch Interaction Recognition**: Analyzes multi-frame physical kinematics (rapid approach, intimate contact distance, inter-body struggle motion variance, followed by high-speed divergence / sprint escape).
3. **Automated Alarm Workflow**:
   - **Audible Siren**: Non-blocking alarm buzzer loop on Windows via `winsound` with cross-platform fallback.
   - **Red Alert Interface**: The application border pulses bright red, accompanied by a prominent warning banner (`⚠ SUSPECTED THEFT DETECTED`), incident ID, and confidence score.
   - **Multi-Stage Evidence Capture**: Captures 4 distinct snapshots per incident (`before.jpg` via rolling buffer, `detected.jpg` at trigger, `after_1s.jpg`, and `after_3s.jpg`).
   - **Auditing & Logging**: Creates timestamped incident evidence folders, JSON manifests (`event.json`), and appends to an audit log (`theft_events.csv`).

---

## 2. System Architecture

```
[ CCTV Video / Webcam / File Stream ]
                 │
                 ▼
     [ Frame Preprocessing ]
      - Aspect-ratio safe scaling (960px)
      - Rolling Ring-Buffer (4 seconds)
                 │
                 ▼
     [ YOLOv8 Object Detector ]
      - Classes: Person (0), Bike (1), Motorbike (3), Backpack (24), Handbag (26), Phone (67)
                 │
                 ▼
     [ ByteTrack Multi-Object Tracker ]
      - Trajectory smoothing & identity persistence
      - Kinematic derivatives: velocity (v_x, v_y), speed |v|, acceleration
      - Carried item spatial association
                 │
                 ▼
     [ Spatial-Temporal Theft Analyzer ]
      - Relative distance derivatives: dD/dt (rapid approach convergence)
      - Proximity interaction window (< 110 px)
      - Motion energy disturbance in overlap region
      - Divergence escape speed ratio (v_escape / v_approach >= 1.75x)
      - Composite suspicion score calculation C in [0.0, 1.0]
                 │
                 ├── [ Confidence >= Threshold (default 0.60) ]
                 │                    │
                 ▼                    ▼
     [ Alert Manager ]         [ Screenshot Manager ]
      - Non-blocking Siren      - Extracts before.jpg from ring-buffer
      - Red Visual Banner       - Saves detected.jpg
      - Cooldown timer          - Schedules after_1s.jpg and after_3s.jpg
      - Alarm Acknowledge       - Generates event.json manifest
                 │                    │
                 └─────────┬──────────┘
                           ▼
                 [ Event Logger & CSV Audit ]
                 [ Tkinter Surveillance Station GUI ]
```

---

## 3. Directory Structure

```
ipcv_mini_project/
├── .venv/                         # Pre-configured Python virtual environment
└── MotionWatch/
    ├── main.py                    # Orchestrator & CLI / GUI entry point
    ├── ui.py                      # Modern Dark Surveillance Desktop GUI
    ├── config.py                  # System thresholds, paths, and settings
    ├── detector.py                # YOLOv8 + ByteTrack object detector
    ├── tracker.py                 # Multi-object kinematic tracker & history
    ├── theft_analyzer.py          # Temporal interaction & snatch-theft engine
    ├── alert_manager.py           # Audio alarm & red visual warning manager
    ├── screenshot_manager.py      # Circular buffer & multi-frame evidence capture
    ├── logger.py                  # CSV auditing & async annotated video writer
    ├── requirements.txt           # Dependency specifications
    ├── README.md                  # System documentation & demo guide
    ├── yolov8n.pt                 # Pretrained YOLOv8 weights
    ├── assets/
    │   └── alarm.wav              # Synthesized security alarm audio
    ├── output/
    │   ├── incidents/             # Evidence capture directories
    │   │   └── event_YYYYMMDD_HHMMSS_<source>/
    │   │       ├── before.jpg     # ~1.5s prior to trigger (approach)
    │   │       ├── detected.jpg   # Trigger frame
    │   │       ├── after_1s.jpg   # +1.0s post-trigger
    │   │       ├── after_3s.jpg   # +3.0s post-trigger
    │   │       └── event.json     # Machine-readable incident metadata
    │   ├── logs/
    │   │   └── theft_events.csv   # Comprehensive incident log
    │   └── annotated_videos/      # Recorded video footage with telemetry
    ├── sample.mp4                 # Test video clip 1
    ├── sample2.mp4                # Test video clip 2
    ├── Mynewsdesk.mp4             # Multi-person test clip 3
    └── MPS Phone Theft - Media Cut - June 2026.mp4  # Police street snatch clip
```

---

## 4. Installation & Setup (Windows & VS Code)

### Step 1: Open Terminal in VS Code
Open VS Code in `c:\Users\kripa\Desktop\ipcv_mini_project`. Open a PowerShell terminal (`Ctrl + \``).

### Step 2: Activate the Virtual Environment
```powershell
& ".\.venv\Scripts\Activate.ps1"
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, or directly use `.\.venv\Scripts\python.exe`)*

### Step 3: Verify Dependencies
The virtual environment is already pre-configured with PyTorch, Torchvision, Ultralytics YOLO, and OpenCV. If needed:
```powershell
pip install -r MotionWatch/requirements.txt
```

---

## 5. How to Launch and Operate MotionWatch

### Launching the Graphical Surveillance Station (Recommended)
From the project root:
```powershell
& ".\.venv\Scripts\python.exe" "MotionWatch/main.py"
```
*(Or `cd MotionWatch; python main.py`)*

### Launching with a Specific Video File Directly
```powershell
& ".\.venv\Scripts\python.exe" "MotionWatch/main.py" --video "MotionWatch/MPS Phone Theft - Media Cut - June 2026.mp4"
```

### Launching with Webcam Input
```powershell
& ".\.venv\Scripts\python.exe" "MotionWatch/main.py" --webcam
```

### Running in Headless / Automated Testing Mode
```powershell
& ".\.venv\Scripts\python.exe" "MotionWatch/main.py" --headless --video "MotionWatch/sample.mp4"
```

---

## 6. Graphical Interface Features & Keyboard Shortcuts

| Control | Shortcut | Function |
|---|---|---|
| **📁 Open Video...** | — | Opens Windows file picker to load any `.mp4`, `.avi`, `.mov`, `.mkv` clip |
| **📹 Webcam** | — | Toggles live camera input (device index `0`) |
| **▶ Start** | — | Begins YOLO tracking and real-time theft analysis |
| **⏸ Pause / Resume** | `Space` | Pauses/resumes frame processing and analysis |
| **⏹ Stop** | `Q` / `Esc` | Stops processing loop and flushes pending evidence |
| **🔕 Acknowledge Alarm** | `A` | Immediately silences the audible siren and dismisses red warning |
| **🚨 Manual Alarm Test** | `T` | Triggers a full test alarm sequence with label `[TEST]` |
| **📂 Open Output Folder**| — | Opens `output/incidents/` in Windows Explorer |
| **Theft Threshold Slider**| — | Dynamically adjusts suspicion sensitivity (0.20 to 0.90) |

---

## 7. How to Run a Complete Demonstration (Demo Script)

1. **Launch the Application**:
   ```powershell
   & ".\.venv\Scripts\python.exe" "MotionWatch/main.py"
   ```
2. **Observe Normal Monitoring**:
   - The station displays `MONITORING ACTIVE` with a green indicator.
   - Bounding boxes appear around tracked individuals with their unique track ID, detected category, and calculated speeds in pixels/frame (e.g. `ID 1 person 3.2px/f`).
   - Trajectory trail lines show recent motion paths.
3. **Demonstrate Automated Snatch / Theft Detection**:
   - As people approach each other, notice the telemetry tracking relative distance.
   - When a rapid interception and high-speed divergence occur:
     1. The window border and canvas outline flash **bright red**.
     2. A loud audible siren plays repeatedly without lagging the video stream.
     3. The warning banner displays:
        ```
        ⚠ SUSPECTED THEFT DETECTED
        Event ID: THEFT_YYYYMMDD_HHMMSS_01 | Confidence: 82.5% | Involved: [1, 2]
        ```
4. **Demonstrate Verification via Manual Test Alert**:
   - Click the **🚨 [TEST] Manual Alarm Test** button or press **`T`**.
   - Notice the system immediately initiates the complete alarm pipeline with clear `[TEST]` labeling so test events are never confused with actual incidents.
5. **Silence the Alarm**:
   - Click **🔕 Acknowledge Alarm** or press **`A`**.
   - The audio siren stops instantly and the interface returns to normal state.
6. **Inspect Evidence & Audits**:
   - Double-click any incident in the table at the bottom of the screen, or click **📂 Open Incidents Folder**.
   - Windows Explorer opens `MotionWatch/output/incidents/<event_folder>/`.
   - Inspect the 4 captured screenshots:
     - `before.jpg`: Scene context ~1.5 seconds before the alert.
     - `detected.jpg`: Exact moment the alert was triggered.
     - `after_1s.jpg`: Scene 1.0 second after the event (escape path).
     - `after_3s.jpg`: Scene 3.0 seconds after the event.
     - `event.json`: Detailed machine-readable incident metadata.
   - Open `MotionWatch/output/logs/theft_events.csv` in Excel or VS Code to review the audit trail.

---

## 8. Technical Honesty & Model Limitations

In modern computer vision, **theft is a semantic legal definition**, not a single static object class:
- **What genuinely works**:
  - **Spatial-temporal kinematics**: Measuring approach speed, intimacy threshold, divergence acceleration, and escape velocity ratios is reliable, fast (runs at 15–30+ FPS on CPU/GPU), and avoids false alarms from benign pedestrians walking alone.
  - **Motion energy perturbation**: Inter-bounding-box pixel differencing detects physical contact/struggle spikes.
  - **ByteTrack ID persistence**: Retains target identities across occlusions and motion blur.
  - **Multi-stage evidence scheduling**: Exact frame-accurate capture of before and after states.
- **Approximations & Limitations**:
  - The model does not read intent. Two friends playfully running towards each other or high-fiving might trigger a high-speed interaction alert if velocity thresholds are exceeded.
  - Heavy video compression, extreme low-light CCTV, or extreme camera perspective distortion (e.g. extreme fisheye) can alter pixel velocity measurements.
  - The sensitivity threshold slider is provided specifically so security operators can tune the trigger point for specific camera angles and distances.
