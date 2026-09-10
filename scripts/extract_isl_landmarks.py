#!/usr/bin/env python3
"""Create normalized pose sequences from an ISL video collection.

The output JSON is compatible with ``src/generation/animate.py`` and records
the source clip, gloss label, frame rate, and MediaPipe tracking coverage.
It is the required data-preparation step before lookup animation or training
a pose-generation model.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
POSE_COUNT, HAND_COUNT = 33, 21


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "untitled"


def landmarks_to_array(landmarks, count: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((count, 3), dtype=np.float32)
    return np.array([[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32)


def normalize_frame(pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Center all landmarks at the shoulder midpoint and scale by shoulder width."""
    left_shoulder, right_shoulder = pose[11], pose[12]
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if shoulder_width < 1e-5:
        return pose, left_hand, right_hand, False
    center = (left_shoulder + right_shoulder) / 2.0
    normalized_left = (left_hand - center) / shoulder_width if np.any(left_hand) else left_hand
    normalized_right = (right_hand - center) / shoulder_width if np.any(right_hand) else right_hand
    return (
        (pose - center) / shoulder_width,
        normalized_left,
        normalized_right,
        True,
    )


def extract_video(video_path: Path, output_path: Path, gloss: str, pose_model: Path, hand_model: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[dict] = []
    tracked_pose_frames = 0
    vision = mp.tasks.vision
    running_mode = vision.RunningMode.VIDEO
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=running_mode,
        num_poses=1,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=running_mode,
        num_hands=2,
    )

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
            vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = round(frame_index * 1000 / fps)
            pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)
            pose = np.zeros((POSE_COUNT, 3), dtype=np.float32)
            if pose_result.pose_landmarks:
                pose = np.array([[point.x, point.y, point.z] for point in pose_result.pose_landmarks[0]], dtype=np.float32)
            left, right = np.zeros((HAND_COUNT, 3), dtype=np.float32), np.zeros((HAND_COUNT, 3), dtype=np.float32)
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name.lower() if handedness else ""
                hand = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
                if label == "left":
                    left = hand
                elif label == "right":
                    right = hand
            pose, left, right, tracked = normalize_frame(pose, left, right)
            tracked_pose_frames += int(tracked)
            frames.append({"pose": pose.tolist(), "left_hand": left.tolist(), "right_hand": right.tolist()})
            frame_index += 1
    cap.release()

    if not frames:
        raise ValueError("Video contains no readable frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "gloss": gloss,
        "source_video": str(video_path),
        "fps": fps,
        "num_frames": len(frames),
        "normalization": "shoulder_midpoint_and_width",
        "tracking": {"pose_frames": tracked_pose_frames, "pose_coverage": tracked_pose_frames / len(frames)},
        "frames": frames,
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normalized MediaPipe landmarks from ISL videos")
    parser.add_argument("--input-dir", required=True, help="Directory containing ISL videos (searched recursively)")
    parser.add_argument("--output-dir", default="data/isl/keypoints", help="Directory for landmark JSON sequences")
    parser.add_argument("--label-from", choices=("parent", "filename"), default="parent", help="How to derive a provisional gloss label")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of videos for a smoke test")
    parser.add_argument("--pose-model", default="models/mediapipe/pose_landmarker_full.task", help="Official MediaPipe Pose Landmarker task model")
    parser.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task", help="Official MediaPipe Hand Landmarker task model")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    videos = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"No supported video files found under {input_dir}")
    pose_model, hand_model = Path(args.pose_model), Path(args.hand_model)
    for model_path in (pose_model, hand_model):
        if not model_path.is_file():
            raise SystemExit(f"Missing MediaPipe task model: {model_path}")

    manifest = []
    for index, video in enumerate(videos, start=1):
        label_source = video.parent.name if args.label_from == "parent" else video.stem
        gloss = safe_name(label_source)
        # Prefixing with the relative path prevents clips with identical names overwriting one another.
        output_name = safe_name(str(video.relative_to(input_dir).with_suffix(""))) + ".json"
        try:
            payload = extract_video(video, output_dir / output_name, gloss, pose_model, hand_model)
            manifest.append({"gloss": gloss, "landmarks": output_name, "source_video": str(video), **payload["tracking"]})
            print(f"[{index}/{len(videos)}] {video.name}: {payload['tracking']['pose_coverage']:.0%} pose coverage")
        except ValueError as error:
            print(f"[{index}/{len(videos)}] Skipped {video.name}: {error}")

    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} landmark sequences and metadata to {output_dir}")


if __name__ == "__main__":
    main()
