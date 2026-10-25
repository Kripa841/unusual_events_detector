"""
MotionWatch - Surveillance Monitoring Station GUI
Tkinter-based dark theme desktop control center with embedded OpenCV video display,
real-time telemetry, interactive controls, and evidence inspector.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from config import (
    UI_THEME,
    OUTPUT_DIR,
    INCIDENTS_DIR,
    DEFAULT_THEFT_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_CONFIDENCE
)


class SurveillanceUI:
    """Tkinter Desktop Control Station for MotionWatch."""

    def __init__(self, controller):
        self.controller = controller
        
        self.root = tk.Tk()
        self.root.title("MotionWatch — CCTV Theft Detection & Alert System")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)
        self.root.configure(bg=UI_THEME["bg_dark"])

        # State variables
        self.tk_image = None
        self.selected_video_name = tk.StringVar(value="No video loaded")
        self.status_text = tk.StringVar(value="STANDBY")
        self.timestamp_text = tk.StringVar(value="--:--:--")
        self.fps_text = tk.StringVar(value="0.0 FPS")
        self.tracks_count_text = tk.StringVar(value="0 Tracks (0 Persons)")
        self.screenshots_count_text = tk.StringVar(value="0 Captured")
        self.events_count_text = tk.StringVar(value="0 Events")
        
        self.theft_thresh_var = tk.DoubleVar(value=DEFAULT_THEFT_CONFIDENCE_THRESHOLD)
        self.det_thresh_var = tk.DoubleVar(value=DEFAULT_DETECTION_CONFIDENCE)
        self.sound_enabled_var = tk.BooleanVar(value=True)
        self.save_video_var = tk.BooleanVar(value=False)

        self._build_styles()
        self._build_layout()
        self._bind_shortcuts()
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Force window to foreground so user immediately sees it
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(1000, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except Exception:
            pass

    def _build_styles(self):
        """Configure ttk styles for dark modern surveillance look."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Base Frame styles
        style.configure("TFrame", background=UI_THEME["bg_dark"])
        style.configure("Card.TFrame", background=UI_THEME["panel_bg"], relief="flat")
        
        # Label styles
        style.configure("TLabel", background=UI_THEME["panel_bg"], foreground=UI_THEME["text_main"], font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=UI_THEME["bg_dark"], foreground="#ffffff", font=("Segoe UI", 14, "bold"))
        style.configure("Muted.TLabel", background=UI_THEME["panel_bg"], foreground=UI_THEME["text_muted"], font=("Segoe UI", 8))
        style.configure("Metric.TLabel", background=UI_THEME["panel_bg"], foreground="#60a5fa", font=("Segoe UI", 11, "bold"))
        
        # Button styles
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.map("Primary.TButton",
            background=[("active", "#1d4ed8"), ("!disabled", "#2563eb")],
            foreground=[("!disabled", "#ffffff")]
        )
        style.map("Alert.TButton",
            background=[("active", "#b91c1c"), ("!disabled", "#ef4444")],
            foreground=[("!disabled", "#ffffff")]
        )
        style.map("Success.TButton",
            background=[("active", "#059669"), ("!disabled", "#10b981")],
            foreground=[("!disabled", "#ffffff")]
        )
        style.map("Secondary.TButton",
            background=[("active", "#374151"), ("!disabled", "#1f2937")],
            foreground=[("!disabled", "#f3f4f6")]
        )

        # Treeview for event logs
        style.configure("Treeview",
            background=UI_THEME["panel_bg"],
            foreground=UI_THEME["text_main"],
            fieldbackground=UI_THEME["panel_bg"],
            rowheight=26,
            font=("Segoe UI", 9)
        )
        style.configure("Treeview.Heading",
            background="#283142",
            foreground="#ffffff",
            font=("Segoe UI", 9, "bold")
        )
        style.map("Treeview", background=[("selected", "#3b82f6")])

    def _build_layout(self):
        """Constructs UI layout: Top Header, Main View (Video + Side Controls), and Bottom Log."""
        # ================= Top Header Bar =================
        header_bar = tk.Frame(self.root, bg=UI_THEME["bg_dark"], height=56)
        header_bar.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(10, 6))

        # Logo and Title
        title_frame = tk.Frame(header_bar, bg=UI_THEME["bg_dark"])
        title_frame.pack(side=tk.LEFT)
        
        badge_lbl = tk.Label(title_frame, text=" CCTV AI ", bg="#2563eb", fg="#ffffff", font=("Segoe UI", 8, "bold"), padx=4, pady=2)
        badge_lbl.pack(side=tk.LEFT, padx=(0, 8))

        app_title = tk.Label(title_frame, text="MotionWatch", fg="#ffffff", bg=UI_THEME["bg_dark"], font=("Segoe UI", 14, "bold"))
        app_title.pack(side=tk.LEFT)

        app_subtitle = tk.Label(title_frame, text="— Autonomous Theft & Snatch Alert System", fg=UI_THEME["text_muted"], bg=UI_THEME["bg_dark"], font=("Segoe UI", 10))
        app_subtitle.pack(side=tk.LEFT, padx=(6, 0))

        # Status and Telemetry Bar (Right)
        telemetry_frame = tk.Frame(header_bar, bg=UI_THEME["bg_dark"])
        telemetry_frame.pack(side=tk.RIGHT)

        self.status_badge = tk.Label(
            telemetry_frame,
            textvariable=self.status_text,
            bg="#374151",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
            relief="flat"
        )
        self.status_badge.pack(side=tk.RIGHT, padx=6)

        fps_lbl = tk.Label(telemetry_frame, textvariable=self.fps_text, bg=UI_THEME["bg_dark"], fg="#9ca3af", font=("Consolas", 10))
        fps_lbl.pack(side=tk.RIGHT, padx=10)

        # ================= Banner Alert Area (Appears on Theft) =================
        self.alert_banner = tk.Frame(self.root, bg="#ef4444", height=44)
        self.alert_banner_lbl = tk.Label(
            self.alert_banner,
            text="⚠ SUSPECTED THEFT DETECTED",
            bg="#ef4444",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold")
        )
        self.alert_banner_lbl.pack(side=tk.LEFT, padx=18, pady=8)

        self.alert_details_lbl = tk.Label(
            self.alert_banner,
            text="",
            bg="#ef4444",
            fg="#fee2e2",
            font=("Segoe UI", 9)
        )
        self.alert_details_lbl.pack(side=tk.LEFT, padx=10, pady=8)

        ack_banner_btn = tk.Button(
            self.alert_banner,
            text="ACKNOWLEDGE ALARM (A)",
            command=self.controller.acknowledge_alarm,
            bg="#ffffff",
            fg="#b91c1c",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=2,
            relief="flat",
            cursor="hand2"
        )
        ack_banner_btn.pack(side=tk.RIGHT, padx=16, pady=6)
        # Initially hidden
        # self.alert_banner.pack(...)

        # ================= Center Content Split =================
        center_frame = tk.Frame(self.root, bg=UI_THEME["bg_dark"])
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=4)

        # Video Canvas Container
        self.video_container = tk.Frame(center_frame, bg="#000000", highlightthickness=2, highlightbackground="#283142")
        self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas = tk.Label(self.video_container, bg="#000000", text="[ No Video Stream ]\nClick 'Open Video File' or 'Use Webcam' to start", fg="#6b7280", font=("Segoe UI", 12))
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Right Side Panel: Controls & Telemetry
        side_panel = tk.Frame(center_frame, bg=UI_THEME["panel_bg"], width=330, highlightthickness=1, highlightbackground=UI_THEME["panel_border"])
        side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        side_panel.pack_propagate(False)

        self._build_side_panel(side_panel)

        # ================= Bottom Logs & Evidence Area =================
        bottom_frame = tk.Frame(self.root, bg=UI_THEME["panel_bg"], height=160, highlightthickness=1, highlightbackground=UI_THEME["panel_border"])
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(6, 12))
        bottom_frame.pack_propagate(False)

        self._build_bottom_panel(bottom_frame)

    def _build_side_panel(self, parent: tk.Frame):
        """Builds controls, parameter sliders, and metrics."""
        pad_x, pad_y = 12, 8

        # --- Section: Video Source ---
        src_lbl = tk.Label(parent, text="VIDEO SOURCE & CONTROLS", font=("Segoe UI", 9, "bold"), fg="#93c5fd", bg=UI_THEME["panel_bg"])
        src_lbl.pack(anchor="w", padx=pad_x, pady=(10, 4))

        # Current loaded file label
        curr_file_lbl = tk.Label(parent, textvariable=self.selected_video_name, fg="#d1d5db", bg=UI_THEME["panel_bg"], font=("Segoe UI", 8), wraplength=300, justify="left")
        curr_file_lbl.pack(anchor="w", padx=pad_x, pady=(0, 6))

        btn_row1 = tk.Frame(parent, bg=UI_THEME["panel_bg"])
        btn_row1.pack(fill=tk.X, padx=pad_x, pady=3)
        
        btn_open = ttk.Button(btn_row1, text="📁 Open Video...", style="Secondary.TButton", command=self.controller.select_video_file)
        btn_open.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        btn_cam = ttk.Button(btn_row1, text="📹 Webcam", style="Secondary.TButton", command=self.controller.select_webcam)
        btn_cam.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Monitoring Playback Controls
        btn_row2 = tk.Frame(parent, bg=UI_THEME["panel_bg"])
        btn_row2.pack(fill=tk.X, padx=pad_x, pady=4)

        self.btn_start = ttk.Button(btn_row2, text="▶ Start", style="Success.TButton", command=self.controller.start_monitoring)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_pause = ttk.Button(btn_row2, text="⏸ Pause", style="Secondary.TButton", command=self.controller.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        self.btn_stop = ttk.Button(btn_row2, text="⏹ Stop", style="Secondary.TButton", command=self.controller.stop_monitoring)
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        # Divider
        tk.Frame(parent, bg=UI_THEME["panel_border"], height=1).pack(fill=tk.X, padx=pad_x, pady=8)

        # --- Section: Alarm & Evidence Actions ---
        act_lbl = tk.Label(parent, text="ALARM & ACTIONS", font=("Segoe UI", 9, "bold"), fg="#93c5fd", bg=UI_THEME["panel_bg"])
        act_lbl.pack(anchor="w", padx=pad_x, pady=(0, 4))

        btn_ack = ttk.Button(parent, text="🔕 Acknowledge Alarm (Key: A)", style="Alert.TButton", command=self.controller.acknowledge_alarm)
        btn_ack.pack(fill=tk.X, padx=pad_x, pady=3)

        btn_test = ttk.Button(parent, text="🚨 [TEST] Manual Alarm Test", style="Secondary.TButton", command=self.controller.trigger_manual_test)
        btn_test.pack(fill=tk.X, padx=pad_x, pady=3)

        btn_folder = ttk.Button(parent, text="📂 Open Incidents Folder", style="Secondary.TButton", command=self.controller.open_output_folder)
        btn_folder.pack(fill=tk.X, padx=pad_x, pady=3)

        # Divider
        tk.Frame(parent, bg=UI_THEME["panel_border"], height=1).pack(fill=tk.X, padx=pad_x, pady=8)

        # --- Section: Real-time Telemetry ---
        tele_lbl = tk.Label(parent, text="LIVE SURVEILLANCE TELEMETRY", font=("Segoe UI", 9, "bold"), fg="#93c5fd", bg=UI_THEME["panel_bg"])
        tele_lbl.pack(anchor="w", padx=pad_x, pady=(0, 4))

        metrics_frame = tk.Frame(parent, bg="#131720", highlightthickness=1, highlightbackground=UI_THEME["panel_border"])
        metrics_frame.pack(fill=tk.X, padx=pad_x, pady=4)

        # Tracks row
        r1 = tk.Frame(metrics_frame, bg="#131720")
        r1.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(r1, text="Active Tracks:", bg="#131720", fg=UI_THEME["text_muted"]).pack(side=tk.LEFT)
        tk.Label(r1, textvariable=self.tracks_count_text, bg="#131720", fg="#60a5fa", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT)

        # Screenshots count row
        r2 = tk.Frame(metrics_frame, bg="#131720")
        r2.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(r2, text="Saved Screenshots:", bg="#131720", fg=UI_THEME["text_muted"]).pack(side=tk.LEFT)
        tk.Label(r2, textvariable=self.screenshots_count_text, bg="#131720", fg="#34d399", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT)

        # Events count row
        r3 = tk.Frame(metrics_frame, bg="#131720")
        r3.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(r3, text="Total Incidents:", bg="#131720", fg=UI_THEME["text_muted"]).pack(side=tk.LEFT)
        tk.Label(r3, textvariable=self.events_count_text, bg="#131720", fg="#f87171", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT)

        # Divider
        tk.Frame(parent, bg=UI_THEME["panel_border"], height=1).pack(fill=tk.X, padx=pad_x, pady=8)

        # --- Section: Sensitivity & Options ---
        opt_lbl = tk.Label(parent, text="DETECTION SENSITIVITY", font=("Segoe UI", 9, "bold"), fg="#93c5fd", bg=UI_THEME["panel_bg"])
        opt_lbl.pack(anchor="w", padx=pad_x, pady=(0, 2))

        # Theft Threshold Slider
        s1_frame = tk.Frame(parent, bg=UI_THEME["panel_bg"])
        s1_frame.pack(fill=tk.X, padx=pad_x, pady=2)
        tk.Label(s1_frame, text="Theft Threshold:", bg=UI_THEME["panel_bg"], fg=UI_THEME["text_muted"], font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.theft_val_lbl = tk.Label(s1_frame, text=f"{self.theft_thresh_var.get():.2f}", bg=UI_THEME["panel_bg"], fg="#60a5fa", font=("Segoe UI", 8, "bold"))
        self.theft_val_lbl.pack(side=tk.RIGHT)

        s1 = ttk.Scale(parent, from_=0.20, to=0.90, variable=self.theft_thresh_var, command=self._on_theft_thresh_change)
        s1.pack(fill=tk.X, padx=pad_x, pady=(0, 6))

        # Checkboxes
        tk.Checkbutton(
            parent,
            text="Enable Audible Siren",
            variable=self.sound_enabled_var,
            command=self._on_sound_toggle,
            bg=UI_THEME["panel_bg"],
            fg=UI_THEME["text_main"],
            selectcolor="#1f2937",
            activebackground=UI_THEME["panel_bg"],
            activeforeground="#ffffff",
            font=("Segoe UI", 8)
        ).pack(anchor="w", padx=pad_x, pady=1)

        tk.Checkbutton(
            parent,
            text="Save Annotated Video to Disk",
            variable=self.save_video_var,
            bg=UI_THEME["panel_bg"],
            fg=UI_THEME["text_main"],
            selectcolor="#1f2937",
            activebackground=UI_THEME["panel_bg"],
            activeforeground="#ffffff",
            font=("Segoe UI", 8)
        ).pack(anchor="w", padx=pad_x, pady=1)

    def _build_bottom_panel(self, parent: tk.Frame):
        """Builds recent incidents audit table and evidence inspector."""
        header_row = tk.Frame(parent, bg=UI_THEME["panel_bg"])
        header_row.pack(fill=tk.X, padx=12, pady=(6, 4))

        tk.Label(header_row, text="RECENT DETECTED INCIDENTS & EVIDENCE LOG", font=("Segoe UI", 9, "bold"), fg="#93c5fd", bg=UI_THEME["panel_bg"]).pack(side=tk.LEFT)

        btn_open_selected = tk.Button(
            header_row,
            text="🔍 Inspect Incident Evidence",
            command=self._open_selected_incident_folder,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=1,
            relief="flat",
            cursor="hand2"
        )
        btn_open_selected.pack(side=tk.RIGHT)

        # Treeview Columns
        columns = ("id", "time", "source", "type", "confidence", "folder", "ack")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=4)
        
        self.tree.heading("id", text="Event ID")
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("source", text="Video Clip")
        self.tree.heading("type", text="Incident Classification")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("folder", text="Incident Evidence Folder")
        self.tree.heading("ack", text="Acknowledged")

        self.tree.column("id", width=190, anchor="w")
        self.tree.column("time", width=140, anchor="center")
        self.tree.column("source", width=140, anchor="w")
        self.tree.column("type", width=180, anchor="w")
        self.tree.column("confidence", width=80, anchor="center")
        self.tree.column("folder", width=340, anchor="w")
        self.tree.column("ack", width=100, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 6))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=(0, 6))

        # Double click handler
        self.tree.bind("<Double-1>", lambda e: self._open_selected_incident_folder())

    def _bind_shortcuts(self):
        """Keyboard shortcuts for fast operator response."""
        self.root.bind("<space>", lambda e: self.controller.toggle_pause())
        self.root.bind("<a>", lambda e: self.controller.acknowledge_alarm())
        self.root.bind("<A>", lambda e: self.controller.acknowledge_alarm())
        self.root.bind("<t>", lambda e: self.controller.trigger_manual_test())
        self.root.bind("<T>", lambda e: self.controller.trigger_manual_test())
        self.root.bind("<q>", lambda e: self.controller.stop_monitoring())
        self.root.bind("<Q>", lambda e: self.controller.stop_monitoring())

    def _on_theft_thresh_change(self, val):
        thresh = float(val)
        self.theft_val_lbl.config(text=f"{thresh:.2f}")
        self.controller.set_theft_threshold(thresh)

    def _on_sound_toggle(self):
        self.controller.set_sound_enabled(self.sound_enabled_var.get())

    def show_alert_banner(self, event_data: Dict[str, Any]):
        """Displays large red alert banner across the UI."""
        eid = event_data.get("event_id", "")
        conf = event_data.get("confidence", 0.0)
        etype = event_data.get("event_type", "SUSPECTED THEFT")
        
        if event_data.get("is_manual_test", False):
            self.alert_banner.config(bg="#f59e0b")
            self.alert_banner_lbl.config(text="[TEST] ⚠ MANUAL TEST ALERT", bg="#f59e0b")
            self.alert_details_lbl.config(text=f"Event ID: {eid} | Verification Mode", bg="#f59e0b")
        else:
            self.alert_banner.config(bg="#ef4444")
            self.alert_banner_lbl.config(text="⚠ SUSPECTED THEFT DETECTED", bg="#ef4444")
            self.alert_details_lbl.config(text=f"Event ID: {eid} | Confidence: {conf*100:.1f}% | Involved: {event_data.get('involved_track_ids')}", bg="#ef4444")

        self.alert_banner.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(0, 6), before=self.root.winfo_children()[2])
        self.status_badge.config(text="⚠ ALERT ACTIVE", bg="#ef4444")
        self.video_container.config(highlightbackground="#ef4444", highlightthickness=4)

    def hide_alert_banner(self):
        """Hides the red alert banner and resets normal style."""
        self.alert_banner.pack_forget()
        is_monitoring = self.controller.is_monitoring
        if is_monitoring:
            self.status_badge.config(text="MONITORING ACTIVE", bg="#10b981")
            self.video_container.config(highlightbackground="#10b981", highlightthickness=2)
        else:
            self.status_badge.config(text="STANDBY", bg="#374151")
            self.video_container.config(highlightbackground="#283142", highlightthickness=2)

    def update_frame(self, bgr_frame: np.ndarray):
        """Updates the video canvas with the latest processed CCTV frame."""
        h, w = bgr_frame.shape[:2]
        
        # Calculate resize maintaining aspect ratio inside canvas
        canvas_w = max(320, self.video_canvas.winfo_width())
        canvas_h = max(240, self.video_canvas.winfo_height())

        scale = min(canvas_w / w, canvas_h / h)
        target_w = int(w * scale)
        target_h = int(h * scale)

        if target_w > 0 and target_h > 0:
            resized = cv2.resize(bgr_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self.tk_image = ImageTk.PhotoImage(image=img)
            self.video_canvas.config(image=self.tk_image, text="")

    def add_event_to_table(self, event_data: Dict[str, Any]):
        """Inserts detected incident record at the top of the GUI audit table."""
        eid = event_data.get("event_id", "")
        tstamp = event_data.get("timestamp", "")
        source = Path(event_data.get("video_source", "Unknown")).name
        etype = event_data.get("event_type", "THEFT")
        conf = f"{event_data.get('confidence', 0.0)*100:.1f}%"
        folder = event_data.get("incident_folder", "")
        ack = "YES" if event_data.get("acknowledged", False) else "NO"

        self.tree.insert("", 0, values=(eid, tstamp, source, etype, conf, folder, ack))
        # Keep total count updated
        current_count = len(self.tree.get_children())
        self.events_count_text.set(f"{current_count} Incidents")

    def load_initial_events(self, events: List[Dict[str, Any]]):
        """Populates the table with historical events from CSV."""
        for ev in events:
            eid = ev.get("Event_ID", "")
            tstamp = ev.get("Timestamp", "")
            source = ev.get("Video_Source", "")
            etype = ev.get("Event_Type", "")
            try:
                cval = float(ev.get("Confidence", 0.0))
                conf = f"{cval*100:.1f}%"
            except Exception:
                conf = "N/A"
            folder = ev.get("Incident_Folder", "")
            ack = "YES" if ev.get("Acknowledged", "").lower() in ["true", "yes"] else "NO"

            self.tree.insert("", tk.END, values=(eid, tstamp, source, etype, conf, folder, ack))
        self.events_count_text.set(f"{len(events)} Incidents")

    def _open_selected_incident_folder(self):
        """Opens the selected event's screenshot evidence folder in Windows Explorer."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Evidence Inspector", "Please select an incident from the table below to inspect screenshots.")
            return

        values = self.tree.item(selected[0], "values")
        folder_path = values[5]
        if folder_path and Path(folder_path).exists():
            if sys.platform == "win32":
                os.startfile(folder_path)
            else:
                subprocess.run(["xdg-open", folder_path])
        else:
            messagebox.showwarning("Folder Not Found", f"Evidence folder could not be found:\n{folder_path}")

    def on_close(self):
        """Safe application termination."""
        self.controller.shutdown()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
