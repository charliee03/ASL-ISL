#!/usr/bin/env python3
"""Extract normalized MediaPipe landmarks from ISL-CSLRT image sequences.

ISL-CSLRT stores each signer rendition as an ordered directory of JPG frames.
This script turns every rendition into one compact JSON sequence and writes a
manifest with its English sentence label.  The output deliberately remains
separate from the INCLUDE-50 word-level keypoints: it is for sentence-level
recognition/translation experiments, not avatar lookup playback.
"""

import argparse
from contextlib import nullcontext
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
POSE_COUNT, HAND_COUNT = 33, 21


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "untitled"


def natural_key(path: Path) -> list[object]:
    """Sort ``frame 2.jpg`` before ``frame 10.jpg``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def normalize_frame(pose: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    left_shoulder, right_shoulder = pose[11], pose[12]
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if shoulder_width < 1e-5:
        return pose, left, right, False
    center = (left_shoulder + right_shoulder) / 2.0
    # Missing hands are represented by exact zeros. Do not translate those
    # zeros into plausible-looking coordinates during normalization.
    normalized_left = (left - center) / shoulder_width if np.any(left) else left
    normalized_right = (right - center) / shoulder_width if np.any(right) else right
    return ((pose - center) / shoulder_width, normalized_left, normalized_right, True)


def landmark_array(landmarks, count: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((count, 3), dtype=np.float32)
    return np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)


def extract_sequence(frame_dir: Path, output_path: Path, sentence: str, signer: str, pose_landmarker, hand_landmarker) -> dict:
    frames = sorted((path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
    if not frames:
        raise ValueError("sequence contains no readable image files")

    output_frames: list[dict] = []
    tracked_pose_frames = 0
    tracked_hand_frames = 0
    for frame_path in frames:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pose_result = pose_landmarker.detect(image)
        hand_result = hand_landmarker.detect(image) if hand_landmarker else None
        tracked_hand_frames += int(bool(hand_result and hand_result.hand_landmarks))
        pose = landmark_array(pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, POSE_COUNT)
        left, right = np.zeros((HAND_COUNT, 3), dtype=np.float32), np.zeros((HAND_COUNT, 3), dtype=np.float32)
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness) if hand_result else ():
            label = handedness[0].category_name.lower() if handedness else ""
            if label == "left":
                left = landmark_array(landmarks, HAND_COUNT)
            elif label == "right":
                right = landmark_array(landmarks, HAND_COUNT)
        pose, left, right, tracked = normalize_frame(pose, left, right)
        tracked_pose_frames += int(tracked)
        output_frames.append({"pose": pose.tolist(), "left_hand": left.tolist(), "right_hand": right.tolist()})
    if not output_frames:
        raise ValueError("sequence contains no decodable images")
    payload = {
        "schema_version": "1.0",
        "feature_schema_version": "2.0",
        "dataset": "ISL-CSLRT",
        "sentence": sentence,
        "signer": signer,
        "source_frames": str(frame_dir),
        "num_frames": len(output_frames),
        "normalization": "shoulder_midpoint_and_width",
        "tracking": {
            "pose_frames": tracked_pose_frames,
            "pose_coverage": tracked_pose_frames / len(output_frames),
            "hand_frames": tracked_hand_frames,
            "hand_coverage": tracked_hand_frames / len(output_frames),
        },
        "frames": output_frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks from ISL-CSLRT JPG sentence sequences")
    parser.add_argument("--input-dir", required=True, help="ISL-CSLRT Frames_Sentence_Level directory")
    parser.add_argument("--output-dir", default="data/isl/cslrt_keypoints", help="Destination for JSON sequences and metadata")
    parser.add_argument("--limit", type=int, help="Process only this many sequences (smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing output files")
    parser.add_argument("--pose-only", action="store_true", help="Extract pose only; use when the two task models cannot fit in memory together")
    parser.add_argument("--pose-model", default="models/mediapipe/pose_landmarker_full.task")
    parser.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    pose_model, hand_model = Path(args.pose_model), Path(args.hand_model)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not pose_model.is_file() or not hand_model.is_file():
        raise SystemExit("Missing MediaPipe task model(s). Run the INCLUDE-50 extractor setup first.")

    sequences = sorted(path for path in input_dir.glob("*/*") if path.is_dir())
    if args.limit is not None:
        sequences = sequences[:args.limit]
    if not sequences:
        raise SystemExit(f"No sentence/signer directories found under {input_dir}")
    vision = mp.tasks.vision
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model)), running_mode=vision.RunningMode.IMAGE, num_poses=1
    )
    hand_options = None if args.pose_only else vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)), running_mode=vision.RunningMode.IMAGE, num_hands=2
    )
    manifest = []
    # Creating MediaPipe Tasks per recording can exhaust memory on long runs.
    # Keep one pair alive for this entire batch instead.
    hand_context = nullcontext(None) if hand_options is None else vision.HandLandmarker.create_from_options(hand_options)
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, hand_context as hand_landmarker:
        for index, frame_dir in enumerate(sequences, start=1):
            sentence, signer = frame_dir.parent.name, frame_dir.name
            output_name = f"{safe_name(sentence)}__signer_{safe_name(signer)}.json"
            output_path = output_dir / output_name
            if output_path.exists() and not args.overwrite:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                print(f"[{index}/{len(sequences)}] Reused {output_name}")
            else:
                try:
                    payload = extract_sequence(frame_dir, output_path, sentence, signer, pose_landmarker, hand_landmarker)
                    print(f"[{index}/{len(sequences)}] {sentence!r}, signer {signer}: {payload['tracking']['pose_coverage']:.0%} pose coverage")
                except ValueError as error:
                    print(f"[{index}/{len(sequences)}] Skipped {frame_dir}: {error}")
                    continue
            manifest.append({"sentence": sentence, "signer": signer, "landmarks": output_name, **payload["tracking"], "num_frames": payload["num_frames"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote metadata for {len(manifest)} sentence sequences to {output_dir}")


if __name__ == "__main__":
    main()
