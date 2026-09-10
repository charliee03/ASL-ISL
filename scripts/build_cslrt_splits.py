#!/usr/bin/env python3
"""Build leakage-free signer-independent splits for ISL-CSLRT landmarks."""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_SPLIT_SIGNERS = {
    "train": {"1", "2", "3", "4", "5"},
    "val": {"6"},
    "test": {"7"},
}


def build_splits(metadata: list[dict], keypoints_dir: Path) -> tuple[dict[str, list[dict]], list[str]]:
    sentences = sorted({row["sentence"].strip() for row in metadata})
    sentence_to_id = {sentence: index for index, sentence in enumerate(sentences)}
    signer_to_split = {
        signer: split
        for split, signers in DEFAULT_SPLIT_SIGNERS.items()
        for signer in signers
    }
    splits = {name: [] for name in DEFAULT_SPLIT_SIGNERS}
    seen_ids: set[str] = set()

    for row in metadata:
        signer = str(row["signer"])
        if signer not in signer_to_split:
            raise ValueError(f"Signer {signer!r} is not assigned to a split")
        sentence = row["sentence"].strip()
        sample_id = f"{sentence_to_id[sentence]:03d}_signer_{signer}"
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        landmark_path = keypoints_dir / row["landmarks"]
        if not landmark_path.is_file():
            raise FileNotFoundError(f"Missing landmark sequence: {landmark_path}")
        payload = json.loads(landmark_path.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        if len(frames) != row["num_frames"]:
            raise ValueError(f"Frame count mismatch in {landmark_path}")
        has_hands = any(
            any(abs(value) > 1e-8 for point in frame.get(side, []) for value in point)
            for frame in frames
            for side in ("left_hand", "right_hand")
        )
        splits[signer_to_split[signer]].append({
            "id": sample_id,
            "sentence": sentence,
            "class_id": sentence_to_id[sentence],
            "signer": signer,
            "landmarks": row["landmarks"],
            "num_frames": row["num_frames"],
            "pose_coverage": row["pose_coverage"],
            "has_detected_hands": has_hands,
        })
    for rows in splits.values():
        rows.sort(key=lambda row: (row["class_id"], int(row["signer"])))
    return splits, sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ISL-CSLRT signer-independent train/val/test manifests")
    parser.add_argument("--keypoints-dir", default="data/isl/cslrt_keypoints")
    parser.add_argument("--output-dir", default="data/isl/cslrt_splits")
    args = parser.parse_args()

    keypoints_dir, output_dir = Path(args.keypoints_dir), Path(args.output_dir)
    metadata_path = keypoints_dir / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing extraction manifest: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    splits, vocabulary = build_splits(metadata, keypoints_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")

    summary = {"strategy": "signer_independent", "keypoints_dir": str(keypoints_dir), "classes": len(vocabulary), "splits": {}}
    for split, rows in splits.items():
        (output_dir / f"{split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        counts = Counter(row["sentence"] for row in rows)
        summary["splits"][split] = {
            "signers": sorted(DEFAULT_SPLIT_SIGNERS[split]),
            "samples": len(rows),
            "classes_present": len(counts),
            "missing_classes": sorted(set(vocabulary) - set(counts)),
            "samples_without_detected_hands": sum(not row["has_detected_hands"] for row in rows),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assigned = [row for rows in splits.values() for row in rows]
    if len(assigned) != len(metadata):
        raise RuntimeError("Not every metadata entry was assigned exactly once")
    print(f"Created signer-independent manifests for {len(assigned)} samples and {len(vocabulary)} sentence classes")
    for split, stats in summary["splits"].items():
        print(f"{split}: {stats['samples']} samples, {stats['classes_present']} classes, signers {', '.join(stats['signers'])}")


if __name__ == "__main__":
    main()
