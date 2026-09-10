#!/usr/bin/env python3
"""Validate extracted MSASL schema-v2 NPZ feature sequences."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_schema_version") != "2.0":
        raise ValueError("MSASL metadata is not feature schema 2.0")
    root = metadata_path.parent
    samples = metadata.get("samples", [])
    if not samples:
        raise ValueError("MSASL metadata contains no samples")
    pose_frames = hand_frames = total_frames = 0
    seen_ids = set()
    for row in samples:
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen_ids.add(row["id"])
        path = root / row["features"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as payload:
            sequence = payload["keypoints"]
        if sequence.ndim != 3 or sequence.shape[1:] != (75, 3):
            raise ValueError(f"Invalid shape in {path}: {sequence.shape}")
        if not np.isfinite(sequence).all() or not np.any(sequence):
            raise ValueError(f"Empty or non-finite features in {path}")
        pose = sequence[:, :33]
        tracked = np.any(pose != 0, axis=(1, 2))
        if np.any(tracked):
            centers = (pose[tracked, 11] + pose[tracked, 12]) / 2
            widths = np.linalg.norm(pose[tracked, 11, :2] - pose[tracked, 12, :2], axis=1)
            if np.max(np.abs(centers)) > 1e-3 or not np.allclose(widths, 1.0, atol=1e-3):
                raise ValueError(f"Invalid shoulder normalization in {path}")
        total_frames += len(sequence)
        pose_frames += int(tracked.sum())
        hand_frames += int(np.any(sequence[:, 33:] != 0, axis=(1, 2)).sum())
    return {
        "samples": len(samples),
        "frames": total_frames,
        "pose_coverage": pose_frames / total_frames,
        "hand_coverage": hand_frames / total_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate extracted MSASL pose/hand features")
    parser.add_argument("--keypoints-dir", default="data/asl/msasl100_keypoints")
    args = parser.parse_args()
    result = validate(Path(args.keypoints_dir) / "metadata.json")
    print(
        f"Samples: {result['samples']}; frames: {result['frames']}; "
        f"pose coverage: {result['pose_coverage']:.2%}; "
        f"hand coverage: {result['hand_coverage']:.2%}"
    )
    print("MSASL landmark schema and normalization are valid")


if __name__ == "__main__":
    main()
