#!/usr/bin/env python3
"""Create deterministic class-stratified splits for CSLRT word images."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints-dir", default="data/isl/cslrt_word_keypoints")
    parser.add_argument("--output-dir", default="data/isl/cslrt_word_splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    keypoints_dir, output_dir = Path(args.keypoints_dir), Path(args.output_dir)
    rows = json.loads((keypoints_dir / "metadata.json").read_text())
    vocabulary = sorted({row["word"] for row in rows})
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    grouped = defaultdict(list)
    for row in rows:
        if not (keypoints_dir / row["landmarks"]).is_file():
            raise FileNotFoundError(row["landmarks"])
        grouped[row["word"]].append(row)
    splits = {"train": [], "val": [], "test": []}
    for word in vocabulary:
        samples = sorted(grouped[word], key=lambda row: row["landmarks"])
        random.Random(f"{args.seed}:{word}").shuffle(samples)
        n = len(samples)
        test_n = 1 if n >= 2 else 0
        val_n = 1 if n >= 3 else 0
        if n >= 10:
            test_n = max(1, round(n * 0.15))
            val_n = max(1, round(n * 0.15))
        for split, partition in (("test", samples[:test_n]), ("val", samples[test_n:test_n + val_n]), ("train", samples[test_n + val_n:])):
            for index, row in enumerate(partition):
                splits[split].append({"id": f"{word_to_id[word]:03d}_{split}_{index:03d}", "word": word,
                    "class_id": word_to_id[word], "landmarks": row["landmarks"],
                    "has_detected_hands": row["hand_coverage"] > 0})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2))
    summary = {"strategy": "deterministic_class_stratified", "seed": args.seed, "classes": len(vocabulary), "splits": {}}
    for split, samples in splits.items():
        samples.sort(key=lambda row: (row["class_id"], row["id"]))
        (output_dir / f"{split}.json").write_text(json.dumps(samples, indent=2))
        present = Counter(row["word"] for row in samples)
        summary["splits"][split] = {"samples": len(samples), "classes_present": len(present), "missing_classes": sorted(set(vocabulary) - set(present))}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Created word splits for {len(rows)} samples and {len(vocabulary)} classes")
    for split, data in summary["splits"].items(): print(f"{split}: {data['samples']} samples, {data['classes_present']} classes")


if __name__ == "__main__":
    main()
