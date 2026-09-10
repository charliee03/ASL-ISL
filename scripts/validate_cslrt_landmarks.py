#!/usr/bin/env python3
"""Validate coordinate consistency and completeness of CSLRT landmarks."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints-dir", default="data/isl/cslrt_keypoints")
    args = parser.parse_args()
    root = Path(args.keypoints_dir)
    files = sorted(path for path in root.glob("*.json") if path.name != "metadata.json")
    if not files:
        raise SystemExit(f"No landmark sequences found in {root}")
    invalid = []
    pose_frames = hand_frames = total_frames = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("feature_schema_version") != "2.0":
            invalid.append(f"{path.name}: old or missing feature schema")
            continue
        for index, frame in enumerate(payload.get("frames", [])):
            pose = np.asarray(frame["pose"], dtype=np.float32)
            left = np.asarray(frame["left_hand"], dtype=np.float32)
            right = np.asarray(frame["right_hand"], dtype=np.float32)
            total_frames += 1
            if pose.shape != (33, 3) or left.shape != (21, 3) or right.shape != (21, 3):
                invalid.append(f"{path.name}:{index}: wrong landmark shape")
                continue
            if not all(np.isfinite(array).all() for array in (pose, left, right)):
                invalid.append(f"{path.name}:{index}: non-finite coordinate")
            if np.any(pose):
                pose_frames += 1
                midpoint = (pose[11] + pose[12]) / 2
                shoulder_width = np.linalg.norm(pose[11, :2] - pose[12, :2])
                if np.linalg.norm(midpoint) > 1e-3 or abs(shoulder_width - 1.0) > 1e-3:
                    invalid.append(f"{path.name}:{index}: pose is not shoulder-normalized")
            if np.any(left) or np.any(right):
                hand_frames += 1
                detected = np.concatenate([array[np.any(array != 0, axis=1)] for array in (left, right)])
                if detected.size and np.max(np.abs(detected)) > 20:
                    invalid.append(f"{path.name}:{index}: implausible normalized hand coordinate")
    pose_coverage = pose_frames / total_frames if total_frames else 0.0
    hand_coverage = hand_frames / total_frames if total_frames else 0.0
    print(f"Sequences: {len(files)}; frames: {total_frames}; pose coverage: {pose_coverage:.2%}; hand coverage: {hand_coverage:.2%}")
    if invalid:
        print("Validation failed:")
        for problem in invalid[:20]:
            print(f"- {problem}")
        raise SystemExit(f"Found {len(invalid)} validation problem(s)")
    print("Landmark schema and normalization are valid")


if __name__ == "__main__":
    main()
