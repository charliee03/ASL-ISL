"""
Re-anchors left_hand / right_hand landmarks to the pose wrist for every frame
in already-generated keypoint JSONs (fixes hand/arm detachment caused by
smoothing pose and hand streams with independent OneEuroFilter instances). 
Utilised after the creation of the gloss json files.

Usage:
    python fix_hand_attachment.py keypoints/            # fix a folder in-place (writes *_fixed.json next to originals)
    python fix_hand_attachment.py keypoints/ --inplace   # overwrite originals
    python fix_hand_attachment.py one_file.json          # fix a single file
"""

import argparse
import json
from pathlib import Path

import numpy as np

# pose landmark indices (MediaPipe Pose/Holistic, world landmarks)
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16


def fix_frame(frame: dict) -> None:
    """Re-anchor hand root (landmark 0) to the corresponding pose wrist,
    in place, preserving the (already-smoothed) hand shape/articulation."""
    if frame.get("pose") is None:
        return

    pose = np.array(frame["pose"])

    for hand_key, wrist_idx in (("left_hand", LEFT_WRIST_IDX), ("right_hand", RIGHT_WRIST_IDX)):
        hand = frame.get(hand_key)
        if hand is None:
            continue
        hand = np.array(hand)
        offset = hand - hand[0]              # finger shape relative to the hand's own root
        hand = pose[wrist_idx] + offset       # re-anchor root to the smoothed pose wrist
        frame[hand_key] = hand.tolist()


def fix_file(path: Path, out_path: Path) -> None:
    with open(path, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        fix_frame(frame)

    with open(out_path, "w") as f:
        json.dump(data, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=str, help="JSON file or folder of JSON files")
    ap.add_argument("--inplace", action="store_true", help="Overwrite original files instead of writing *_fixed.json")
    args = ap.parse_args()

    target = Path(args.target)
    files = [target] if target.is_file() else sorted(target.glob("*.json"))

    if not files:
        print("No JSON files found.")
        return

    for path in files:
        out_path = path if args.inplace else path.with_name(path.stem + "_fixed.json")
        fix_file(path, out_path)
        print(f"Fixed: {path.name} -> {out_path.name}")

    print(f"\nDone. {len(files)} file(s) processed.")


if __name__ == "__main__":
    main()
