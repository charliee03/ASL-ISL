#!/usr/bin/env python3
"""Build official, quality-filtered train/val/test manifests for MSASL features."""

import argparse
import json
from collections import Counter
from pathlib import Path


def build_splits(metadata: dict, min_pose_coverage: float, min_hand_coverage: float):
    rows = metadata.get("samples", [])
    class_ids = sorted({int(row["class_id"]) for row in rows})
    if class_ids and class_ids != list(range(max(class_ids) + 1)):
        raise ValueError("MSASL class IDs must be contiguous from zero")
    vocabulary = [None] * (max(class_ids) + 1 if class_ids else 0)
    splits = {"train": [], "val": [], "test": []}
    rejected = Counter()
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen.add(row["id"])
        class_id = int(row["class_id"])
        if vocabulary[class_id] not in (None, row["gloss"]):
            raise ValueError(f"Conflicting gloss for class {class_id}")
        vocabulary[class_id] = row["gloss"]
        if row["split"] not in splits:
            rejected["unknown_split"] += 1
        elif row["pose_coverage"] < min_pose_coverage:
            rejected["low_pose_coverage"] += 1
        elif row["hand_coverage"] < min_hand_coverage:
            rejected["low_hand_coverage"] += 1
        else:
            splits[row["split"]].append(row)
    if any(value is None for value in vocabulary):
        raise ValueError("At least one class has no extracted samples")
    for rows_for_split in splits.values():
        rows_for_split.sort(key=lambda row: (int(row["class_id"]), row["id"]))
    signer_sets = {
        split: {str(row["signer"]) for row in rows_for_split}
        for split, rows_for_split in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = signer_sets[left] & signer_sets[right]
        if overlap:
            raise ValueError(f"Signer leakage between {left} and {right}: {sorted(overlap)}")
    return splits, vocabulary, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality-filtered MSASL official splits")
    parser.add_argument("--keypoints-dir", default="data/asl/msasl100_keypoints")
    parser.add_argument("--output-dir", default="data/asl/msasl100_splits")
    parser.add_argument("--min-pose-coverage", type=float, default=0.80)
    parser.add_argument("--min-hand-coverage", type=float, default=0.10)
    parser.add_argument("--drop-classes-missing-from-splits", action="store_true")
    args = parser.parse_args()
    keypoints_dir, output_dir = Path(args.keypoints_dir), Path(args.output_dir)
    metadata = json.loads((keypoints_dir / "metadata.json").read_text(encoding="utf-8"))
    splits, vocabulary, rejected = build_splits(
        metadata, args.min_pose_coverage, args.min_hand_coverage
    )
    dropped_classes = []
    if args.drop_classes_missing_from_splits:
        represented = [set(int(row["class_id"]) for row in rows) for rows in splits.values()]
        eligible = set.intersection(*represented)
        dropped_classes = [vocabulary[index] for index in range(len(vocabulary)) if index not in eligible]
        remap = {old: new for new, old in enumerate(sorted(eligible))}
        vocabulary = [vocabulary[old] for old in sorted(eligible)]
        for split, rows in splits.items():
            filtered = []
            for row in rows:
                if int(row["class_id"]) in remap:
                    row = dict(row)
                    row["class_id"] = remap[int(row["class_id"])]
                    filtered.append(row)
            splits[split] = filtered
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    summary = {
        "strategy": "official_msasl_signer_independent_splits_with_quality_filter",
        "quality_thresholds": {
            "min_pose_coverage": args.min_pose_coverage,
            "min_hand_coverage": args.min_hand_coverage,
        },
        "classes": len(vocabulary),
        "dropped_classes_missing_from_splits": dropped_classes,
        "rejected": dict(rejected),
        "splits": {},
    }
    for split, rows in splits.items():
        (output_dir / f"{split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        represented = {int(row["class_id"]) for row in rows}
        summary["splits"][split] = {
            "samples": len(rows),
            "classes_present": len(represented),
            "missing_classes": [vocabulary[index] for index in range(len(vocabulary)) if index not in represented],
            "signers": sorted({str(row["signer"]) for row in rows}),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Built MSASL manifests for {sum(len(rows) for rows in splits.values())} accepted samples")
    for split, stats in summary["splits"].items():
        print(f"{split}: {stats['samples']} samples, {stats['classes_present']} classes, {len(stats['signers'])} signers")
    print(f"Rejected: {dict(rejected)}")


if __name__ == "__main__":
    main()
