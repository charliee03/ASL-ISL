"""Verify the three implemented ASL-video to recorded-ISL-pose demo paths.

Run with ``AITE_ENABLE_RECOGNITION=true``. This is an evidence-producing smoke
test, not a replacement for held-out evaluation or ISL signer review.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch

from src.api import server

DEMO_CASES = {
    "hello": "Dataset/MS-ASL/videos/class_0_signer_26_22302.mp4",
    "drink": "Dataset/MS-ASL/videos/class_56_signer_10_22073.mp4",
    "man": "Dataset/MS-ASL/videos/class_58_signer_10_24288.mp4",
}


def extract_and_predict(video_path: Path) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as output:
        feature_path = Path(output.name)
    try:
        command = [
            str(PROJECT_DIR / ".venv-cslrt/bin/python"),
            str(PROJECT_DIR / "scripts/extract_msasl_single_video.py"),
            "--input-video", str(video_path),
            "--output-file", str(feature_path),
            "--num-frames", str(server.NUM_FRAMES),
        ]
        result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "feature extraction failed")
        with np.load(feature_path) as payload:
            keypoints = payload["keypoints"].astype(np.float32)
            tracking = json.loads(str(payload["tracking_json"].item()))
        with torch.no_grad():
            probabilities = torch.softmax(
                server.model(torch.from_numpy(keypoints).unsqueeze(0).to(server.device)), dim=1
            )[0]
            confidence, class_index = probabilities.max(dim=0)
        gloss = server.gloss_vocab[class_index.item()]
        return {
            "gloss": gloss,
            "confidence": float(confidence),
            "accepted": float(confidence) >= server.RECOGNITION_MIN_CONFIDENCE,
            "tracking": tracking,
        }
    finally:
        feature_path.unlink(missing_ok=True)


async def verify_case(expected_gloss: str, relative_video: str) -> dict:
    video_path = PROJECT_DIR / relative_video
    prediction = extract_and_predict(video_path)
    if prediction["gloss"] != expected_gloss or not prediction["accepted"]:
        return {"video": relative_video, "expected_gloss": expected_gloss, **prediction, "passed": False}
    translation = await server.translate_text({"asl_gloss": prediction["gloss"]})
    avatar = await server.generate_avatar({"isl_gloss": translation["isl_gloss"]})
    generated_path = server.GENERATED_DIR / avatar["video_url"].rsplit("/", 1)[-1]
    try:
        passed = avatar["mode"] == "landmark_playback" and generated_path.is_file() and generated_path.stat().st_size > 0
        return {
            "video": relative_video,
            "expected_gloss": expected_gloss,
            **prediction,
            "isl_gloss": translation["isl_gloss"],
            "avatar_mode": avatar["mode"],
            "avatar_source_gloss": avatar["source_gloss"],
            "passed": passed,
        }
    finally:
        generated_path.unlink(missing_ok=True)


async def main_async(output_path: Path) -> None:
    if server.model is None:
        raise RuntimeError("Set AITE_ENABLE_RECOGNITION=true before running this verification.")
    results = [await verify_case(gloss, video) for gloss, video in DEMO_CASES.items()]
    report = {
        "scope": "Three implemented ASL isolated-sign to recorded ISL-pose paths only.",
        "recognition_threshold": server.RECOGNITION_MIN_CONFIDENCE,
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/demo_video_to_avatar_verification.json"))
    args = parser.parse_args()
    asyncio.run(main_async(args.output))


if __name__ == "__main__":
    main()
