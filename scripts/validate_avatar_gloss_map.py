#!/usr/bin/env python3
"""Validate that every configured avatar alias points to an extracted landmark class."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="configs/avatar_gloss_map.json")
    parser.add_argument("--metadata", default="data/isl/keypoints/metadata.json")
    args = parser.parse_args()
    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    available = {entry["gloss"] for entry in metadata}
    missing = sorted({target for target in mapping["aliases"].values() if target not in available})
    if missing:
        raise SystemExit(f"Aliases point to missing landmark classes: {', '.join(missing)}")
    print(f"Valid: {len(mapping['aliases'])} aliases map to {len(available)} extracted landmark classes.")
    print(f"Status: {mapping.get('status', 'unspecified')}")


if __name__ == "__main__":
    main()
