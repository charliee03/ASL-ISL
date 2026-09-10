#!/usr/bin/env python3
"""Report experimental INCLUDE-50 candidate recognition metrics by class.

This script intentionally reports recognition evidence only.  It never treats an
English label overlap as evidence of an ASL-to-ISL mapping or changes API routes.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize experimental candidate-class metrics")
    parser.add_argument("--results", required=True, help="Evaluator JSON containing per_class metrics")
    parser.add_argument("--candidates", required=True, help="JSON list of proposed English-label candidates")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    per_class = results.get("per_class", {})
    report = {
        "purpose": "Recognition evidence for candidate selection; not ASL-to-ISL linguistic validation.",
        "source_results": str(Path(args.results)),
        "split": results.get("split"),
        "classes_in_checkpoint": results.get("classes_in_checkpoint"),
        "candidate_count_requested": len(candidates),
        "candidate_count_evaluated": sum(name in per_class for name in candidates),
        "candidates": [],
    }
    for name in candidates:
        metrics = per_class.get(name)
        row = {
            "english_label": name,
            "recognition_status": "evaluated" if metrics else "not_in_all_quality-filtered_splits",
            "playback_status": "blocked_pending_ISL_linguistic_review_and_recorded_motion_approval",
        }
        if metrics:
            row.update({key: metrics[key] for key in ("support", "precision", "recall", "f1")})
        report["candidates"].append(row)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {report['candidate_count_evaluated']}/{report['candidate_count_requested']} "
        f"candidate metrics to {args.output}"
    )


if __name__ == "__main__":
    main()
