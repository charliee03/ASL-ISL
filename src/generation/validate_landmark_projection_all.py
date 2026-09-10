"""
===========================================================================
File: validate_landmark_projection_all.py

Description:
    Visualizes MediaPipe pose and hand landmarks stored in JSON files
    alongside the corresponding source videos.

    The script reconstructs a 2D skeleton from the exported landmark
    coordinates and displays it next to the original video frame.

    Optionally, the combined visualization can be saved as a video.

    Visualization:
        - White : Body pose
        - Green : Left hand
        - Red   : Right hand

    Usage:
        Visualization only:
            python validate_landmark_projection_all.py <keypoints> <videos>

        Visualization + save:
            python validate_landmark_projection_all.py <keypoints> <videos> --output <output_folder>
===========================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ================================================================
# Configuration & Constants
# ================================================================

DISPLAY_WIDTH = 1400
SCALE = 700
Y_OFFSET = 120

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ================================================================
# Helper Functions
# ================================================================

def project(pt, w, h):
    """Projects normalized 3D coordinates to 2D pixel space."""
    x, y, z = pt
    u = int(w / 2 + x * SCALE)
    v = int(h / 2 + y * SCALE + Y_OFFSET)
    return (u, v)

def draw_landmarks(canvas, landmarks, color, w, h):
    """Draws keypoints as circles on the canvas."""
    for p in landmarks:
        cv2.circle(canvas, project(p, w, h), 4, color, -1)

def draw_connections(canvas, landmarks, connections, color, w, h):
    """Draws lines between connected keypoints."""
    for a, b in connections:
        pa = project(landmarks[a], w, h)
        pb = project(landmarks[b], w, h)
        cv2.line(canvas, pa, pb, color, 2)

def find_matching_video(json_path, video_dir):
    """Finds a corresponding video file for a given JSON path."""
    extensions = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]
    for ext in extensions:
        candidate = video_dir / (json_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


# ================================================================
# Main Execution
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate projected landmarks against source videos.")
    parser.add_argument("keypoints_folder", help="Folder containing landmark JSON files.")
    parser.add_argument("videos_folder", help="Folder containing source videos.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional folder to save validated videos. If omitted, videos are only displayed.")
    args = parser.parse_args()

    # Input Paths
    json_dir = Path(args.keypoints_folder)
    video_dir = Path(args.videos_folder)
    output_dir = Path(args.output) if args.output else None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Validate Input Paths
    if not json_dir.is_dir():
        print(f"\nERROR: Keypoints path is not a valid folder: {json_dir}")
        sys.exit(1)
    if not video_dir.is_dir():
        print(f"\nERROR: Videos path is not a valid folder: {video_dir}")
        sys.exit(1)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"\nERROR: No JSON files found in: {json_dir}")
        sys.exit(1)

    # Setup OpenCV Window
    window_name = "Keypoint Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Process Every Sign
    for idx, json_path in enumerate(json_files):
        video_path = find_matching_video(json_path, video_dir)
        if not video_path:
            print(f"Skipping {json_path.stem}: Matching video not found.")
            continue

        print("\n" + "=" * 60)
        print(f"Showing Sign {idx + 1}/{len(json_files)}")
        print(f"JSON : {json_path.name}")
        print(f"VIDEO: {video_path.name}")
        
        if output_dir:
            print(f"OUTPUT: {output_dir / (json_path.stem + '_validated.mp4')}")
        else:
            print("OUTPUT: Not saving")
        print("=" * 60)

        # Load JSON data
        with open(json_path, "r") as f:
            data = json.load(f)
        frames = data.get("frames", [])

        # Open Source Video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        # Optional Video Writer setup
        writer = None
        if output_dir:
            output_path = output_dir / f"{json_path.stem}_validated.mp4"
            output_width = w * 2  # Side-by-side view
            output_height = h
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
            if not writer.isOpened():
                print(f"WARNING: Could not create output video: {output_path}")
                writer = None

        # Resize Window
        display_height = int(DISPLAY_WIDTH * h / (2 * w))
        cv2.resizeWindow(window_name, DISPLAY_WIDTH, display_height)

        # Process Frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(frames):
                break

            # Create Keypoint Canvas
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            frame_data = frames[frame_idx]
            pose = frame_data.get("pose", [])
            left = frame_data.get("left_hand", [])
            right = frame_data.get("right_hand", [])

            # Draw Body (White)
            if pose:
                draw_connections(canvas, pose, POSE_CONNECTIONS, (255, 255, 255), w, h)
                draw_landmarks(canvas, pose, (255, 255, 255), w, h)

            # Draw Left Hand (Green)
            if left:
                draw_connections(canvas, left, HAND_CONNECTIONS, (0, 255, 0), w, h)
                draw_landmarks(canvas, left, (0, 255, 0), w, h)

            # Draw Right Hand (Red)
            if right:
                draw_connections(canvas, right, HAND_CONNECTIONS, (0, 0, 255), w, h)
                draw_landmarks(canvas, right, (0, 0, 255), w, h)

            # Combine Original + Keypoints
            combined = cv2.hconcat([frame, canvas])

            if writer:
                writer.write(combined)

            # Display
            display_frame = cv2.resize(combined, (DISPLAY_WIDTH, int(DISPLAY_WIDTH * combined.shape[0] / combined.shape[1])))
            cv2.imshow(window_name, display_frame)

            # Keyboard interrupt
            if cv2.waitKey(30) & 0xFF == ord("q"):
                cap.release()
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                print("\nExited by user.")
                sys.exit(0)

            frame_idx += 1

        # Release resources for current video
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {output_path}")

        # Transition to Next Sign
        if idx != len(json_files) - 1:
            next_name = json_files[idx + 1].stem
            print(f"\n{'=' * 60}\nFinished : {json_path.stem}\nNext Sign: {next_name}")
            print("Starting in 2 seconds...\n" + "=" * 60)
            cv2.waitKey(2000)

    # Final Cleanup
    cv2.destroyAllWindows()
    print("\n==========================================")
    print("All sign/video pairs have been completed.")
    if output_dir:
        print(f"Validated videos saved to: {output_dir}")
    else:
        print("No videos were saved (output folder was not provided).")
    print("==========================================")


if __name__ == "__main__":
    main()