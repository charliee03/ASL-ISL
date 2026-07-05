#!/usr/bin/env python3
"""Export gloss vocabulary from WLASL annotation files.

Replicates the vocabulary-building logic from WLASLDataset (NSLT dict branch)
in src/recognition/dataset.py so that the translation module can load the
mapping without instantiating the full dataset.
"""

import json
import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    annotation_file = project_root / "data" / "wlasl" / "nslt_100.json"
    class_list_file = project_root / "data" / "wlasl" / "wlasl_class_list.txt"
    output_file = project_root / "models" / "recognition" / "gloss_vocab.json"

    # ---- Load class list (id -> gloss) ----
    # Mirrors WLASLDataset: line.strip().split(), parts[0]=id, parts[1]=gloss
    class_id_to_gloss: dict[int, str] = {}
    if not class_list_file.exists():
        print(f"ERROR: class list not found: {class_list_file}", file=sys.stderr)
        sys.exit(1)

    with open(class_list_file) as clf:
        for line in clf:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    c_id = int(parts[0])
                    gloss = parts[1]
                    class_id_to_gloss[c_id] = gloss
                except ValueError:
                    pass

    # ---- Load annotations and collect glosses (all splits) ----
    if not annotation_file.exists():
        print(f"ERROR: annotation file not found: {annotation_file}", file=sys.stderr)
        sys.exit(1)

    with open(annotation_file) as f:
        data = json.load(f)

    classes: set[str] = set()
    for video_id, info in data.items():
        action = info.get("action", [])
        if len(action) > 0:
            class_id = action[0]
            gloss = class_id_to_gloss.get(class_id, f"class_{class_id}")
            classes.add(gloss)

    # Sort alphabetically — same as WLASLDataset
    sorted_classes = sorted(classes)

    # ---- Build mappings ----
    gloss_to_id = {gloss: idx for idx, gloss in enumerate(sorted_classes)}
    id_to_gloss = {str(idx): gloss for idx, gloss in enumerate(sorted_classes)}

    vocab = {
        "version": "1.0",
        "dataset": "WLASL-100",
        "num_classes": len(sorted_classes),
        "gloss_to_id": gloss_to_id,
        "id_to_gloss": id_to_gloss,
    }

    # ---- Write output ----
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Wrote {output_file}  ({vocab['num_classes']} classes)")
    # Quick sanity check: print first 5 entries
    for i in range(min(5, len(sorted_classes))):
        print(f"  {i}: {sorted_classes[i]}")
    if len(sorted_classes) > 5:
        print(f"  ... ({len(sorted_classes) - 5} more)")


if __name__ == "__main__":
    main()
