#!/usr/bin/env python3
"""Add hand landmarks to pose-only ISL-CSLRT keypoint sequences.

Run after ``extract_cslrt_landmarks.py --pose-only`` on machines that cannot
hold PoseLandmarker and HandLandmarker in memory at the same time.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HAND_COUNT = 21

def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]

def hand_array(landmarks):
    return np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Add ISL-CSLRT hand landmarks using one MediaPipe model")
    parser.add_argument("--output-dir", default="data/isl/cslrt_keypoints")
    parser.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task")
    args = parser.parse_args()
    output_dir, hand_model = Path(args.output_dir), Path(args.hand_model)
    files = sorted(path for path in output_dir.glob("*.json") if path.name != "metadata.json")
    if not files or not hand_model.is_file():
        raise SystemExit("Missing keypoint JSON files or HandLandmarker model")
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)), running_mode=vision.RunningMode.IMAGE, num_hands=2)
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for index, path in enumerate(files, 1):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("hand_landmarks_extracted"):
                print(f"[{index}/{len(files)}] Reused {path.name}")
                continue
            images = sorted((p for p in Path(payload["source_frames"]).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
            for frame, image_path in zip(payload["frames"], images):
                bgr = cv2.imread(str(image_path))
                left = right = np.zeros((HAND_COUNT, 3), dtype=np.float32)
                if bgr is not None:
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    result = landmarker.detect(image)
                    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                        if handedness and handedness[0].category_name.lower() == "left": left = hand_array(landmarks)
                        elif handedness: right = hand_array(landmarks)
                frame["left_hand"], frame["right_hand"] = left.tolist(), right.tolist()
            payload["hand_landmarks_extracted"] = True
            path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            print(f"[{index}/{len(files)}] Added hands to {path.name}")

if __name__ == "__main__":
    main()
