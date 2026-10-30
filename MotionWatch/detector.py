"""
MotionWatch - Object Detector & ByteTrack Integration Module
Uses Ultralytics YOLOv8 for real-time person, vehicle, and item detection with tracking.
"""

import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    DETECT_CLASSES,
    DEFAULT_DETECTION_CONFIDENCE,
    CONF_PERSON,
    CONF_ITEM,
    CONF_VEHICLE,
    PERSON_CLASS_ID,
    ITEM_CLASS_IDS,
    VEHICLE_CLASS_IDS
)


class YOLODetector:
    """Wrapper around YOLOv8 model for multi-object detection and ByteTrack tracking."""

    def __init__(self, model_path: str = MODEL_PATH, conf: float = DEFAULT_DETECTION_CONFIDENCE):
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        print(f"[Detector] Initializing YOLOv8 on device: {self.device}...")
        self.model = YOLO(model_path)
        self.names = self.model.names  # Class ID -> Class name mapping
        print(f"[Detector] Model loaded successfully ({len(self.names)} classes available).")

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[Dict[str, Any]]:
        """
        Runs YOLO tracking on a single frame using ByteTrack.
        Returns a list of detected/tracked object dictionaries:
        [
            {
                "track_id": int,
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "is_person": bool,
                "is_item": bool,
                "is_vehicle": bool
            }, ...
        ]
        """
        # Use low base threshold (0.15) for YOLO to prevent dropping small bags/suitcases prematurely
        inference_conf = min(self.conf, 0.15)
        results = self.model.track(
            frame,
            persist=persist,
            tracker="bytetrack.yaml",
            classes=DETECT_CLASSES,
            conf=inference_conf,
            imgsz=640,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, confidences, class IDs, and track IDs
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        # ByteTrack track IDs may be None if not tracked yet
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(coords)

        for i in range(len(coords)):
            x1, y1, x2, y2 = coords[i]
            cid = classes[i]
            cname = self.names.get(cid, f"class_{cid}")
            conf_val = float(confs[i])
            tid = int(track_ids[i])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            is_person = (cid == PERSON_CLASS_ID)
            is_item = (cid in ITEM_CLASS_IDS)
            is_vehicle = (cid in VEHICLE_CLASS_IDS)

            # Category-aware confidence filtering ensures bags & suitcases are not dropped
            if is_person and conf_val < CONF_PERSON:
                continue
            elif is_item and conf_val < CONF_ITEM:
                continue
            elif is_vehicle and conf_val < CONF_VEHICLE:
                continue
            elif not (is_person or is_item or is_vehicle) and conf_val < self.conf:
                continue

            detections.append({
                "track_id": tid,
                "class_id": cid,
                "class_name": cname,
                "confidence": conf_val,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": (float(cx), float(cy)),
                "is_person": is_person,
                "is_item": is_item,
                "is_vehicle": is_vehicle
            })

        return detections

    def set_confidence(self, conf: float):
        """Dynamically update detection confidence threshold."""
        self.conf = max(0.1, min(0.95, conf))
