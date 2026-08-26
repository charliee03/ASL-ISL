"""Run twenty real MS-ASL clips through the deployed API and write a QA CSV report.

Start the server first, for example:
    .venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8000
Then run:
    .venv/bin/python scripts/run_qa.py
"""
import asyncio
import argparse
import csv
import io
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from starlette.datastructures import UploadFile

DATASET_DIR = PROJECT_DIR / "Dataset/MS-ASL"
API_URL = "http://127.0.0.1:8000"
REPORT_PATH = PROJECT_DIR / "qa_report.csv"


def post_multipart(url: str, video: Path) -> dict:
    boundary = "----AITEQABoundary"
    content_type = mimetypes.guess_type(video.name)[0] or "video/mp4"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{video.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + video.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def post_json(url: str, data: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def select_clips() -> list[tuple[Path, str]]:
    annotations = json.loads((DATASET_DIR / "MSASL_unified.json").read_text())
    selected, labels = [], set()
    for record in annotations:
        clip = DATASET_DIR / record["video"]
        label = record["label"]
        if label not in labels and clip.is_file() and clip.stat().st_size > 0:
            selected.append((clip, record["gloss"].upper()))
            labels.add(label)
        if len(selected) == 20:
            return selected
    raise RuntimeError(f"Only found {len(selected)} usable unique-class clips; need 20")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Number of selected clips to run (default: 20)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        raise SystemExit("--limit must be between 1 and 20")
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=10) as response:
            health = json.loads(response.read())
        if not health.get("model_loaded"):
            raise SystemExit("Recognition checkpoint is not loaded; QA cannot report genuine predictions.")
        use_http = True
    except urllib.error.URLError:
        # CI and restricted sandboxes may not permit a loopback listener. This
        # calls the same FastAPI endpoint functions and still executes all
        # recognition, translation, and avatar-generation stages.
        from src.api import server
        if server.model is None:
            raise SystemExit("Recognition checkpoint is not loaded; QA cannot report genuine predictions.")
        use_http = False
        print("Loopback server unavailable; running the FastAPI endpoint functions in-process.")
    rows = []
    for number, (clip, expected_gloss) in enumerate(select_clips()[:args.limit], start=1):
        print(f"[{number:02d}/{args.limit}] Starting {clip.name}", flush=True)
        started = time.perf_counter()
        if use_http:
            prediction = post_multipart(f"{API_URL}/predict-sequence", clip)
        else:
            upload = UploadFile(filename=clip.name, file=io.BytesIO(clip.read_bytes()))
            prediction = asyncio.run(server.predict_sequence(upload))
        predicted_gloss = prediction.get("gloss", "")
        if use_http:
            translation = post_json(f"{API_URL}/translate", {"asl_gloss": predicted_gloss})
            avatar = post_json(f"{API_URL}/generate-avatar", {"isl_gloss": translation.get("isl_gloss", "")})
        else:
            translation = asyncio.run(server.translate_gloss({"asl_gloss": predicted_gloss}))
            avatar = asyncio.run(server.generate_avatar({"isl_gloss": translation.get("isl_gloss", "")}))
        latency = time.perf_counter() - started
        rows.append({
            "Test ID": number,
            "Input Video": str(clip.relative_to(PROJECT_DIR)),
            "Source Dataset Gloss": expected_gloss,
            "Predicted ASL": predicted_gloss,
            "Translated ISL": translation.get("isl_gloss", ""),
            "Avatar Generated": bool(avatar.get("video_url")),
            "End-to-End Latency (s)": f"{latency:.3f}",
            "Result": "PASS" if predicted_gloss and translation.get("isl_gloss") and avatar.get("video_url") else "FAIL",
        })
        print(f"[{number:02d}/20] {clip.name}: {rows[-1]['Result']} ({latency:.3f}s)")
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"QA report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
