"""
===========================================================================
File: validate_landmark_projection_all.py

Description:
    Visualizes MediaPipe pose and hand landmarks stored in a JSON file
    alongside the corresponding source video for verification for all in the folders.

    The script reconstructs a 2D skeleton from the exported landmark
    coordinates and displays it next to the original video frame. This
    allows visual inspection of:

        • Correct landmark extraction
        • Left/right hand assignment
        • Skeleton connectivity
        • Coordinate system orientation
        • Scaling and projection accuracy
        • Frame-to-frame temporal consistency

    The visualization uses:
        - White  : Body pose
        - Green  : Left hand
        - Red    : Right hand

    This script is intended as a debugging and validation tool before
    downstream processing such as key-frame extraction, motion smoothing,
    bone rotation estimation, and Blender avatar animation.
===========================================================================
"""

import cv2
import json
import numpy as np
from pathlib import Path
import time

# ---------------- Paths ---------------- #

JSON_DIR = Path("keypoints")
VIDEO_DIR = Path("videos")

DISPLAY_WIDTH = 1400
SCALE = 700
Y_OFFSET = 120

# ---------------- Connections ---------------- #

POSE_CONNECTIONS = [
    (11,13), (13,15),
    (12,14), (14,16),
    (11,12),
    (11,23), (12,24),
    (23,24),
    (23,25), (25,27),
    (24,26), (26,28),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ------------------------------------------------ #

def project(pt, W, H):
    x, y, z = pt
    u = int(W/2 + x * SCALE)
    v = int(H/2 + y * SCALE + Y_OFFSET)
    return (u, v)


def draw_landmarks(canvas, landmarks, color, W, H):
    for p in landmarks:
        cv2.circle(canvas, project(p, W, H), 4, color, -1)


def draw_connections(canvas, landmarks, connections, color, W, H):
    for a, b in connections:
        pa = project(landmarks[a], W, H)
        pb = project(landmarks[b], W, H)
        cv2.line(canvas, pa, pb, color, 2)


# ------------------------------------------------ #

json_files = sorted(JSON_DIR.glob("*.json"))

if not json_files:
    print("No JSON files found.")
    exit()

window_name = "Keypoint Validation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

for idx, json_path in enumerate(json_files):

    video_path = VIDEO_DIR / (json_path.stem + ".MOV")

    if not video_path.exists():
        print(f"Skipping {json_path.stem}: Matching video not found.")
        continue

    print("\n" + "="*60)
    print(f"Showing Sign {idx+1}/{len(json_files)}")
    print(f"JSON : {json_path.name}")
    print(f"VIDEO: {video_path.name}")
    print("="*60)

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    cap = cv2.VideoCapture(str(video_path))

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.resizeWindow(
        window_name,
        DISPLAY_WIDTH,
        int(DISPLAY_WIDTH * H / (2 * W))
    )

    frame_idx = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret or frame_idx >= len(frames):
            break

        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        frame_data = frames[frame_idx]

        pose = frame_data["pose"]
        left = frame_data["left_hand"]
        right = frame_data["right_hand"]

        draw_connections(canvas, pose, POSE_CONNECTIONS, (255,255,255), W, H)
        draw_landmarks(canvas, pose, (255,255,255), W, H)

        draw_connections(canvas, left, HAND_CONNECTIONS, (0,255,0), W, H)
        draw_landmarks(canvas, left, (0,255,0), W, H)

        draw_connections(canvas, right, HAND_CONNECTIONS, (0,0,255), W, H)
        draw_landmarks(canvas, right, (0,0,255), W, H)

        combined = cv2.hconcat([frame, canvas])

        combined = cv2.resize(
            combined,
            (
                DISPLAY_WIDTH,
                int(DISPLAY_WIDTH * combined.shape[0] / combined.shape[1])
            )
        )

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("\nExited by user.")
            exit()

        frame_idx += 1

    cap.release()

    # ---------- Between signs ---------- #

    if idx != len(json_files) - 1:

        next_name = json_files[idx + 1].stem

        print("\n")
        print("="*60)
        print(f"Finished : {json_path.stem}")
        print(f"Next Sign: {next_name}")
        print("Starting in 2 seconds...")
        print("="*60)

        # Keep the last frame visible for 2 seconds
        cv2.waitKey(2000)

cv2.destroyAllWindows()

print("\n==========================================")
print("All sign/video pairs have been completed.")
print("==========================================")