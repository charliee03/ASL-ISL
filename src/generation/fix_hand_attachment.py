"""
Re-anchors left_hand / right_hand landmarks to the pose wrist for every frame
in keypoint JSONs. Fixes hand detachment, scale explosion, corrupt joint deformities,
and high-frequency landmark jitter using anatomical template restoration and 5-frame
Gaussian temporal smoothing.

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


def get_ref_hand_info(frames: list, hand_key: str):
    """Computes median anatomical palm size and canonical normalized finger shape
    across valid tracked frames to restore corrupted or lost hand tracking."""
    valid_offsets_norm = []
    valid_palm_sizes = []

    for f in frames:
        if not f.get("pose"):
            continue
        h = f.get(hand_key)
        if h and len(h) == 21:
            h_arr = np.array(h)
            sz = float(np.linalg.norm(h_arr[9] - h_arr[0]))
            max_reach = float(np.max(np.linalg.norm(h_arr - h_arr[0], axis=1)))
            if 0.035 < sz < 0.22 and max_reach < 0.40:
                valid_offsets_norm.append((h_arr - h_arr[0]) / sz)
                valid_palm_sizes.append(sz)

    if not valid_offsets_norm:
        # Fallback default canonical palm layout if no valid frames detected
        return None, 0.10

    canon_norm = np.median(np.array(valid_offsets_norm), axis=0)
    ref_palm_sz = float(np.median(valid_palm_sizes))
    return canon_norm, ref_palm_sz


def fix_file(path: Path, out_path: Path) -> None:
    with open(path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return

    for hand_key, wrist_idx in (("left_hand", LEFT_WRIST_IDX), ("right_hand", RIGHT_WRIST_IDX)):
        canon_norm, ref_palm_sz = get_ref_hand_info(frames, hand_key)
        if canon_norm is None:
            continue

        raw_offsets_seq = []

        # 1. Clean individual frame hand scale and shape
        for f in frames:
            if not f.get("pose"):
                raw_offsets_seq.append(canon_norm * ref_palm_sz)
                continue

            h = f.get(hand_key)
            if h and len(h) == 21:
                h_arr = np.array(h)
                cur_sz = float(np.linalg.norm(h_arr[9] - h_arr[0]))
                max_reach = float(np.max(np.linalg.norm(h_arr - h_arr[0], axis=1)))

                # Check if hand is corrupted, lost, or exploding
                if cur_sz < 0.04 or cur_sz > 0.22 or max_reach > 3.0 * ref_palm_sz or max_reach < 0.01:
                    clean_offset = canon_norm * ref_palm_sz
                else:
                    scale = ref_palm_sz / cur_sz if (cur_sz > ref_palm_sz * 1.25 or cur_sz < ref_palm_sz * 0.75) else 1.0
                    clean_offset = (h_arr - h_arr[0]) * scale

            else:
                clean_offset = canon_norm * ref_palm_sz

            raw_offsets_seq.append(clean_offset)

        raw_offsets_seq = np.array(raw_offsets_seq)  # Shape: (N_frames, 21, 3)
        N = len(raw_offsets_seq)
        smoothed_seq = np.zeros_like(raw_offsets_seq)

        # 2. Apply 5-frame Gaussian temporal smoothing filter
        weights = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
        for i in range(N):
            acc = np.zeros((21, 3))
            w_sum = 0.0
            for k, offset_k in enumerate([-2, -1, 0, 1, 2]):
                idx = i + offset_k
                if 0 <= idx < N:
                    acc += weights[k] * raw_offsets_seq[idx]
                    w_sum += weights[k]
            smoothed_seq[i] = acc / w_sum

        # 3. Re-anchor to pose wrist
        frame_idx = 0
        for f in frames:
            if f.get("pose"):
                pose = np.array(f["pose"])
                f[hand_key] = (pose[wrist_idx] + smoothed_seq[frame_idx]).tolist()
                frame_idx += 1

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


