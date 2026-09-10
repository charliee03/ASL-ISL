#!/usr/bin/env python3
"""Extract schema-v2 landmarks from ISL-CSLRT word-level still images.

These are isolated-image samples for auxiliary static-sign recognition. They
must not be mixed with the sentence sequence dataset without an explicit
multitask training design.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from extract_cslrt_landmarks import HAND_COUNT, POSE_COUNT, landmark_array, normalize_frame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "untitled"


def main():
    parser = argparse.ArgumentParser(description="Extract ISL-CSLRT word-image pose and hand landmarks")
    parser.add_argument("--input-dir", required=True, help="Frames_Word_Level directory")
    parser.add_argument("--output-dir", default="data/isl/cslrt_word_keypoints")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pose-model", default="models/mediapipe/pose_landmarker_full.task")
    parser.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    pose_model, hand_model = Path(args.pose_model), Path(args.hand_model)
    images = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        images = images[:args.limit]
    if not images or not pose_model.is_file() or not hand_model.is_file():
        raise SystemExit("Missing input images or MediaPipe task model(s)")
    vision = mp.tasks.vision
    pose_options = vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model)), running_mode=vision.RunningMode.IMAGE, num_poses=1)
    hand_options = vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)), running_mode=vision.RunningMode.IMAGE, num_hands=2)
    manifest = []
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        for index, image_path in enumerate(images, 1):
            word = image_path.parent.name
            relative_source = str(image_path.relative_to(input_dir))
            # Some word folders contain repeated filenames. A path digest keeps
            # every source image distinct instead of silently overwriting it.
            source_id = hashlib.sha1(relative_source.encode("utf-8")).hexdigest()[:12]
            source_stem = safe_name(str(image_path.relative_to(input_dir).with_suffix("")))
            output_name = f"{source_stem}__{source_id}.json"
            output_path = output_dir / output_name
            if output_path.exists() and not args.overwrite:
                payload = json.loads(output_path.read_text())
            else:
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    print(f"[{index}/{len(images)}] Skipped unreadable {image_path.name}")
                    continue
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                pose_result, hand_result = pose_landmarker.detect(image), hand_landmarker.detect(image)
                pose = landmark_array(pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, POSE_COUNT)
                left = right = np.zeros((HAND_COUNT, 3), dtype=np.float32)
                for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    if handedness and handedness[0].category_name.lower() == "left":
                        left = landmark_array(landmarks, HAND_COUNT)
                    elif handedness:
                        right = landmark_array(landmarks, HAND_COUNT)
                pose, left, right, tracked = normalize_frame(pose, left, right)
                payload = {"schema_version": "1.0", "feature_schema_version": "2.0", "dataset": "ISL-CSLRT-word-level", "word": word,
                    "source_image": str(image_path), "num_frames": 1, "normalization": "shoulder_midpoint_and_width",
                    "tracking": {"pose_frames": int(tracked), "pose_coverage": float(tracked), "hand_frames": int(bool(hand_result.hand_landmarks)), "hand_coverage": float(bool(hand_result.hand_landmarks))},
                    "frames": [{"pose": pose.tolist(), "left_hand": left.tolist(), "right_hand": right.tolist()}]}
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, separators=(",", ":")))
            manifest.append({"word": word, "landmarks": output_name, **payload["tracking"]})
            print(f"[{index}/{len(images)}] {word}: {payload['tracking']['pose_coverage']:.0%} pose")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} isolated-word landmarks to {output_dir}")


if __name__ == "__main__":
    main()
