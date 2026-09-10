"""Export bounded, secret-free paper metrics and artifact identity metadata."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper_evidence"
MODELS = (
    "msasl100_pose_balanced_large_recognition",
    "msasl126_include50_candidate_recognition",
    "cslrt_recognition_v3",
    "cslrt_word_recognition",
    "isl_motion_autoencoder",
)


def scalars(value):
    """Keep scalar summaries/nested metric objects, excluding example arrays."""
    if isinstance(value, dict):
        return {key: scalars(item) for key, item in value.items()
                if not isinstance(item, list) and key not in {"config", "checkpoint", "path", "per_class"}}
    return value


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    metrics, paths, missing = {}, set(), []
    for model in MODELS:
        folder = ROOT / "models" / model
        metrics[model] = {}
        for name in ("val_results.json", "test_results.json", "calibration.json",
                     "evaluation.json", "runtime.json"):
            path = folder / name
            if path.is_file():
                metrics[model][name] = scalars(json.loads(path.read_text()))
        if not folder.is_dir():
            missing.append(str(folder.relative_to(ROOT)))
        else:
            paths.update(p for p in folder.iterdir() if p.is_file()
                         and p.suffix in {".json", ".yaml", ".csv", ".pt"})
        if not (folder / "best_model.pt").is_file():
            missing.append(str((folder / "best_model.pt").relative_to(ROOT)))
    for pattern in ("configs/*", "src/**/*.py", "src/web/*.js", "src/web/*.html",
                    "src/web/*.css", "scripts/*.py", "tests/*.py",
                    "data/asl/*splits/*.json", "data/isl/*splits/*.json"):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    paths.add(ROOT / "requirements.txt")
    for name in ("IEEE_PAPER_HANDOFF.md", "TRANSLATION.md", "main.tex",
                 "paper_sources.bib", "ASL_ISL_MAPPING_REVIEW_TEMPLATE.csv",
                 "ASL_ISL_MAPPING_REVIEW_GUIDE.md", "DEMO_RUNBOOK.md",
                 "DEMO_VALIDATION_LOG_TEMPLATE.csv", "paper_evidence/VERIFICATION.md"):
        path = ROOT / "docs" / name
        if path.is_file():
            paths.add(path)
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": git("rev-parse", "HEAD"),
                "git_status_porcelain": git("status", "--porcelain").splitlines(),
                "missing_expected_artifacts": missing, "files": []}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["files"].append({"path": str(path.relative_to(ROOT)),
                                  "bytes": path.stat().st_size,
                                  "sha256": digest.hexdigest()})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics.json", metrics), ("manifest.json", manifest)):
        (OUT / name).write_text(json.dumps(value, indent=2) + "\n")
    with (OUT / "recognition_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["model", "split", "num_samples", "top1_accuracy",
                         "top5_accuracy", "macro_f1"])
        for model, summaries in metrics.items():
            for split in ("val", "test"):
                row = summaries.get(f"{split}_results.json")
                if row:
                    writer.writerow([model, split, row.get("samples", row.get("num_samples", "")),
                                     row.get("top1_accuracy", row.get("accuracy", "")),
                                     row.get("top5_accuracy", ""), row.get("macro_f1", "")])
    print(f"Exported {len(manifest['files'])} artifact hashes and {len(metrics)} model summaries.")
    if missing:
        print("Missing expected artifacts: " + ", ".join(missing))


if __name__ == "__main__":
    main()
