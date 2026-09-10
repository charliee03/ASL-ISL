#!/usr/bin/env python3
"""Print the exact MediaPipe startup checkpoint reached before a hard kill."""
from pathlib import Path

def checkpoint(message: str) -> None:
    print(message, flush=True)

checkpoint("1. Python started")
import numpy as np
checkpoint(f"2. NumPy imported ({np.__version__})")
import cv2
checkpoint(f"3. OpenCV imported ({cv2.__version__})")
checkpoint("4. Importing MediaPipe")
import mediapipe as mp
checkpoint(f"5. MediaPipe imported ({mp.__version__})")

model = Path("models/mediapipe/pose_landmarker_full.task")
checkpoint(f"6. Pose model exists: {model.is_file()} ({model.stat().st_size if model.exists() else 0} bytes)")
vision = mp.tasks.vision
options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)
checkpoint("7. Creating PoseLandmarker")
with vision.PoseLandmarker.create_from_options(options):
    checkpoint("8. PoseLandmarker created and closed successfully")
