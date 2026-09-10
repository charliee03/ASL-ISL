#!/usr/bin/env python3
"""Extract one MSASL-schema sequence for upload inference.

Run this helper with ``.venv-cslrt`` so the API process does not need to load
MediaPipe's native runtime.  Its output schema is identical to the training
extractor and can therefore be passed directly to the trained classifier.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from extract_msasl_landmarks import create_landmarkers, extract_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one uploaded ASL video")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--pose-model", default="models/mediapipe/pose_landmarker_full.task")
    parser.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task")
    args = parser.parse_args()

    input_path = Path(args.input_video)
    output_path = Path(args.output_file)
    if not input_path.is_file():
        raise SystemExit(f"Input video does not exist: {input_path}")
    if args.num_frames < 2:
        raise SystemExit("--num-frames must be at least 2")

    pose, hands = create_landmarkers(args.pose_model, args.hand_model)
    try:
        sequence, tracking = extract_video(input_path, args.num_frames, pose, hands)
    finally:
        pose.close()
        hands.close()

    if not np.any(sequence):
        raise SystemExit("No body or hand landmarks were detected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            keypoints=sequence,
            tracking_json=np.asarray(json.dumps(tracking)),
        )
    print(json.dumps(tracking))


if __name__ == "__main__":
    main()
