#!/usr/bin/env python3
"""Create leakage-safe iSign manifests grouped by source video ID.

This prepares metadata only. It does not download gated iSign files and does
not claim that ISL-English supervision is ASL-to-ISL supervision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def bucket(video_id: str) -> str:
    """Stable 80/10/10 allocation by video ID, never by individual segment."""
    value = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "val" if value < 90 else "test"


def parse_uid(uid: str) -> tuple[str, str | None]:
    """Handle both documented iSign IDs and older ISLTranslate source IDs.

    iSign v1.1 documents ``video_id-sequence`` IDs. The public ISLTranslate
    metadata also contains ``video_id_suffix`` rows and full-video IDs with no
    segment suffix. In every case the source-video portion is what must remain
    group-disjoint.
    """
    if "_" in uid:
        video_id, sequence = uid.rsplit("_", 1)
        return video_id, sequence
    if "-" in uid and uid.rsplit("-", 1)[1].isdigit():
        video_id, sequence = uid.rsplit("-", 1)
        return video_id, sequence
    return uid, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="iSign/ISLTranslate uid,text CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = (row.get("uid") or "").strip()
            text = (row.get("text") or "").strip()
            if not uid or not text:
                continue
            video_id, sequence = parse_uid(uid)
            rows.append({"uid": uid, "video_id": video_id, "sequence": sequence, "text": text})
    if not rows:
        raise ValueError("No valid uid,text rows found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        splits[bucket(row["video_id"])].append(row)
    for name, split_rows in splits.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(split_rows, indent=2) + "\n", encoding="utf-8")
    summary = {
        "purpose": "Group-disjoint iSign metadata split; not ASL-to-ISL parallel data.",
        "input_csv": str(args.input_csv),
        "rows": len(rows),
        "unique_video_ids": len({row["video_id"] for row in rows}),
        "splits": {name: {"rows": len(split_rows), "video_ids": len({row["video_id"] for row in split_rows})}
                   for name, split_rows in splits.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
