#!/usr/bin/env python3
"""Rebuild the MSASL extraction manifest from validated NPZ files.

This recovery tool is intentionally independent of extraction.  It repairs a
partial/interrupted ``metadata.json`` without reprocessing thousands of videos.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild MSASL feature metadata from NPZ files")
    parser.add_argument("--data-root", default="Dataset/MS-ASL")
    parser.add_argument("--annotation-file", default="Dataset/MS-ASL/MSASL_unified.json")
    parser.add_argument("--output-dir", default="data/asl/msasl100_keypoints")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()

    data_root, annotation_path, output_dir = (
        Path(args.data_root), Path(args.annotation_file), Path(args.output_dir)
    )
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    selected = [
        row for row in annotations
        if int(row["label"]) < args.num_classes and (data_root / row["video"]).is_file()
    ]
    selected.sort(key=lambda row: (row["split"], int(row["label"]), row["video"]))
    samples, failures = [], []
    expected_shape = (args.num_frames, 75, 3)
    for row in selected:
        video_path = data_root / row["video"]
        feature_path = output_dir / f"{video_path.stem}.npz"
        failure = {
            "id": video_path.stem,
            "video": row["video"],
            "gloss": row["gloss"],
            "class_id": int(row["label"]),
            "signer": str(row.get("signer_id", "unknown")),
            "split": row["split"],
        }
        if not feature_path.is_file():
            failures.append({**failure, "error": "feature file is missing"})
            continue
        try:
            with np.load(feature_path) as payload:
                sequence = payload["keypoints"]
                tracking = json.loads(str(payload["tracking_json"].item()))
            if sequence.shape != expected_shape or not np.isfinite(sequence).all() or not np.any(sequence):
                raise ValueError(f"invalid feature shape or values: {sequence.shape}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failures.append({**failure, "error": str(error)})
            continue
        samples.append({
            **failure,
            "features": feature_path.name,
            "num_frames": args.num_frames,
            **tracking,
        })
    metadata = {
        "schema_version": "1.0",
        "feature_schema_version": "2.0",
        "dataset": f"MSASL-{args.num_classes}",
        "feature_layout": {"pose": 33, "left_hand": 21, "right_hand": 21, "coordinates": 3},
        "normalization": "shoulder_midpoint_and_width",
        "num_frames": args.num_frames,
        "samples": samples,
        "failures": failures,
    }
    output_path = output_dir / "metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} recovered samples and {len(failures)} failures to {output_path}")


if __name__ == "__main__":
    main()
