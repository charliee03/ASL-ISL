"""Inventory potential ASL source data for INCLUDE-50 ISL playback targets.

An exact English-label match is only a data-availability candidate. It is not
evidence that the ASL and ISL signs are equivalent; expert review is required
before a candidate is enabled in the translation or avatar maps.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalise_label(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("Dataset/MS-ASL/MSASL_unified.json"))
    parser.add_argument("--avatar-map", type=Path, default=Path("configs/avatar_gloss_map.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/ASL_ISL_PLAYBACK_CANDIDATES.json"))
    args = parser.parse_args()

    rows = json.loads(args.annotations.read_text(encoding="utf-8"))
    aliases = json.loads(args.avatar_map.read_text(encoding="utf-8")).get("aliases", {})
    targets = sorted({str(target).lower() for target in aliases.values()})

    clips_by_gloss: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        clips_by_gloss[normalise_label(str(row["gloss"]))][str(row.get("split", "unknown"))] += 1

    connected_targets = {"hello", "drink", "man"}
    candidates = []
    for target in targets:
        source_label = normalise_label(target)
        split_counts = clips_by_gloss.get(source_label, Counter())
        if target in connected_targets:
            status = "implemented_unreviewed"
        elif split_counts:
            status = "exact_english_label_candidate_requires_isl_review"
        else:
            status = "no_exact_msasl_source_label"
        candidates.append(
            {
                "include50_target": target,
                "normalised_source_label": source_label,
                "msasl_split_counts": dict(sorted(split_counts.items())),
                "msasl_total_clips": sum(split_counts.values()),
                "status": status,
                "enablement_requirement": (
                    "ISL-expert review, ASL feature extraction, signer-independent evaluation, and an approved mapping"
                    if target not in connected_targets
                    else "ISL-expert review and continued held-out monitoring"
                ),
            }
        )

    report = {
        "purpose": "Data-availability candidates only; not linguistic equivalence evidence.",
        "source_annotations": str(args.annotations),
        "include50_targets": len(targets),
        "implemented_paths": len(connected_targets),
        "exact_english_label_candidates": sum(
            item["status"] == "exact_english_label_candidate_requires_isl_review" for item in candidates
        ),
        "targets_without_exact_msasl_label": sum(
            item["status"] == "no_exact_msasl_source_label" for item in candidates
        ),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {report['implemented_paths']} implemented, "
        f"{report['exact_english_label_candidates']} review candidates, "
        f"{report['targets_without_exact_msasl_label']} without exact MSASL labels."
    )


if __name__ == "__main__":
    main()
