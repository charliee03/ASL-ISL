#!/usr/bin/env python3
"""Extract validated pose and hand features from local MSASL video clips.

This extractor is intentionally separate from the legacy 27-keypoint cache.
It stores 33 pose + 21 left-hand + 21 right-hand landmarks in one normalized
coordinate system and writes a reproducible manifest for MSASL-100 by default.
Run it with the Python 3.12 ``.venv-cslrt`` environment.
"""

import argparse
import json
import re
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_cslrt_landmarks import HAND_COUNT, POSE_COUNT, landmark_array, normalize_frame


_WORKER_POSE = None
_WORKER_HAND = None


def sample_indices(frame_count: int, requested_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("video contains no frames")
    return np.linspace(0, frame_count - 1, requested_frames, dtype=int)


def extract_video(video_path: Path, num_frames: int, pose_landmarker, hand_landmarker) -> tuple[np.ndarray, dict]:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = sample_indices(frame_count, num_frames)
    feature_width = POSE_COUNT + 2 * HAND_COUNT
    features = [np.zeros((feature_width, 3), dtype=np.float32) for _ in range(num_frames)]
    target_positions: dict[int, list[int]] = {}
    for output_position, frame_index in enumerate(indices.tolist()):
        target_positions.setdefault(frame_index, []).append(output_position)
    pose_frames = hand_frames = decoded_frames = 0
    try:
        source_index = 0
        last_target = int(indices[-1])
        while source_index <= last_target:
            ok, bgr = capture.read()
            if not ok:
                break
            positions = target_positions.get(source_index)
            source_index += 1
            if not positions:
                continue
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )
            pose_result = pose_landmarker.detect(image)
            hand_result = hand_landmarker.detect(image)
            pose = landmark_array(
                pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None,
                POSE_COUNT,
            )
            left = np.zeros((HAND_COUNT, 3), dtype=np.float32)
            right = np.zeros((HAND_COUNT, 3), dtype=np.float32)
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name.lower() if handedness else ""
                if label == "left":
                    left = landmark_array(landmarks, HAND_COUNT)
                elif label == "right":
                    right = landmark_array(landmarks, HAND_COUNT)
            pose, left, right, pose_tracked = normalize_frame(pose, left, right)
            combined = np.concatenate((pose, left, right), axis=0)
            for output_position in positions:
                features[output_position] = combined
            repeats = len(positions)
            decoded_frames += repeats
            pose_frames += int(pose_tracked) * repeats
            hand_frames += int(np.any(left) or np.any(right)) * repeats
    finally:
        capture.release()

    sequence = np.asarray(features, dtype=np.float32)
    if sequence.shape != (num_frames, POSE_COUNT + 2 * HAND_COUNT, 3):
        raise ValueError(f"unexpected feature shape {sequence.shape}")
    tracking = {
        "decoded_frames": decoded_frames,
        "decoded_coverage": decoded_frames / num_frames,
        "pose_frames": pose_frames,
        "pose_coverage": pose_frames / num_frames,
        "hand_frames": hand_frames,
        "hand_coverage": hand_frames / num_frames,
        "source_frame_count": frame_count,
        "source_fps": source_fps if source_fps > 0 else None,
    }
    return sequence, tracking


def create_landmarkers(pose_model: str, hand_model: str):
    vision = mp.tasks.vision
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    ))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    ))
    return pose, hand


def init_worker(pose_model: str, hand_model: str) -> None:
    global _WORKER_POSE, _WORKER_HAND
    _WORKER_POSE, _WORKER_HAND = create_landmarkers(pose_model, hand_model)


def process_entry(task: tuple) -> tuple[dict | None, str, dict | None]:
    index, total, row, data_root_value, output_dir_value, num_frames, overwrite = task
    data_root, output_dir = Path(data_root_value), Path(output_dir_value)
    video_path = data_root / row["video"]
    output_name = f"{video_path.stem}.npz"
    output_path = output_dir / output_name
    try:
        if output_path.is_file() and not overwrite:
            with np.load(output_path) as saved:
                sequence = saved["keypoints"]
                tracking = json.loads(str(saved["tracking_json"].item()))
            if sequence.shape != (num_frames, POSE_COUNT + 2 * HAND_COUNT, 3):
                raise ValueError(f"invalid cached shape {sequence.shape}")
            action = "Reused"
        else:
            if _WORKER_POSE is None or _WORKER_HAND is None:
                raise RuntimeError("MediaPipe worker was not initialized")
            sequence, tracking = extract_video(
                video_path, num_frames, _WORKER_POSE, _WORKER_HAND
            )
            with output_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    keypoints=sequence,
                    tracking_json=np.asarray(json.dumps(tracking)),
                )
            action = "Extracted"
    except (OSError, ValueError, cv2.error) as error:
        failure = {
            "id": video_path.stem,
            "video": row["video"],
            "gloss": row["gloss"],
            "class_id": int(row["label"]),
            "signer": str(row.get("signer_id", "unknown")),
            "split": row["split"],
            "error": str(error),
        }
        return None, f"[{index}/{total}] Skipped {video_path.stem}: {error}", failure
    sample_id = video_path.stem
    record = {
        "id": sample_id,
        "video": row["video"],
        "features": output_name,
        "gloss": row["gloss"],
        "class_id": int(row["label"]),
        "signer": str(row.get("signer_id", "unknown")),
        "split": row["split"],
        "num_frames": num_frames,
        **tracking,
    }
    message = (
        f"[{index}/{total}] {action} {sample_id}: "
        f"pose={tracking['pose_coverage']:.0%}, hands={tracking['hand_coverage']:.0%}"
    )
    return record, message, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract schema-v2 MSASL pose and hand landmarks")
    parser.add_argument("--data-root", default="Dataset/MS-ASL")
    parser.add_argument("--annotation-file", default="Dataset/MS-ASL/MSASL_unified.json")
    parser.add_argument("--output-dir", default="data/asl/msasl100_keypoints")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--glosses-file", help="JSON list of MSASL glosses to extract instead of label IDs")
    parser.add_argument("--base-vocabulary", help="Existing vocabulary JSON used to assign appended contiguous class IDs")
    parser.add_argument("--append-metadata", action="store_true", help="Merge new records into existing metadata.json")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1, help="Parallel MediaPipe workers")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based offset after deterministic selection")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pose-model", default="models/mediapipe/pose_landmarker_full.task")
    parser.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotation_path = Path(args.annotation_file)
    output_dir = Path(args.output_dir)
    pose_model, hand_model = Path(args.pose_model), Path(args.hand_model)
    if not data_root.is_dir() or not annotation_path.is_file():
        raise SystemExit("MSASL data root or annotation file is missing")
    if not pose_model.is_file() or not hand_model.is_file():
        raise SystemExit("MediaPipe pose/hand task model is missing")

    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if args.glosses_file:
        requested = json.loads(Path(args.glosses_file).read_text(encoding="utf-8"))
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise SystemExit("--glosses-file must contain a JSON list of gloss strings")
        normalise = lambda value: " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())
        requested_by_normalised = {normalise(gloss): gloss for gloss in requested}
        if len(requested_by_normalised) != len(requested):
            raise SystemExit("--glosses-file contains duplicate normalised glosses")
        base_vocabulary = []
        if args.base_vocabulary:
            base_vocabulary = json.loads(Path(args.base_vocabulary).read_text(encoding="utf-8"))
            if not isinstance(base_vocabulary, list) or not all(isinstance(item, str) for item in base_vocabulary):
                raise SystemExit("--base-vocabulary must contain a JSON list of gloss strings")
        class_ids = {normalise(gloss): index for index, gloss in enumerate(base_vocabulary)}
        for gloss in requested:
            class_ids.setdefault(normalise(gloss), len(class_ids))
        selected, found_glosses = [], set()
        for row in annotations:
            key = normalise(str(row["gloss"]))
            if key in requested_by_normalised and (data_root / row["video"]).is_file():
                remapped = dict(row)
                remapped["label"] = class_ids[key]
                selected.append(remapped)
                found_glosses.add(key)
        missing = sorted(set(requested_by_normalised) - found_glosses)
        if missing:
            print(f"Warning: no local videos found for requested glosses: {', '.join(missing)}")
    else:
        selected = [
            row for row in annotations
            if int(row["label"]) < args.num_classes and (data_root / row["video"]).is_file()
        ]
    selected.sort(key=lambda row: (row["split"], int(row["label"]), row["video"]))
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.limit is not None:
        selected = selected[args.start_index:args.start_index + args.limit]
    elif args.start_index:
        selected = selected[args.start_index:]
    if not selected:
        raise SystemExit("No matching local MSASL videos were found")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (index, len(selected), row, str(data_root), str(output_dir), args.num_frames, args.overwrite)
        for index, row in enumerate(selected, start=1)
    ]
    manifest, failures = [], []
    if args.workers == 1:
        init_worker(str(pose_model), str(hand_model))
        results = map(process_entry, tasks)
        for record, message, failure in results:
            if record is not None:
                manifest.append(record)
            if failure is not None:
                failures.append(failure)
            print(message, flush=True)
        _WORKER_POSE.close()
        _WORKER_HAND.close()
    else:
        context = get_context("spawn")
        with context.Pool(
            args.workers,
            initializer=init_worker,
            initargs=(str(pose_model), str(hand_model)),
        ) as pool:
            for record, message, failure in pool.imap(process_entry, tasks):
                if record is not None:
                    manifest.append(record)
                if failure is not None:
                    failures.append(failure)
                print(message, flush=True)

    previous_samples, previous_failures = [], []
    metadata_path = output_dir / "metadata.json"
    if args.append_metadata and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_samples = list(previous.get("samples", []))
        previous_failures = list(previous.get("failures", []))
    merged_samples = {row["id"]: row for row in previous_samples}
    merged_samples.update({row["id"]: row for row in manifest})
    merged_failures = {row["id"]: row for row in previous_failures}
    merged_failures.update({row["id"]: row for row in failures})
    metadata = {
        "schema_version": "1.0",
        "feature_schema_version": "2.0",
        "dataset": f"MSASL-{len({row['class_id'] for row in merged_samples.values()})}",
        "feature_layout": {"pose": 33, "left_hand": 21, "right_hand": 21, "coordinates": 3},
        "normalization": "shoulder_midpoint_and_width",
        "num_frames": args.num_frames,
        "samples": sorted(merged_samples.values(), key=lambda row: (row["split"], row["class_id"], row["id"])),
        "failures": sorted(merged_failures.values(), key=lambda row: row["id"]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} extracted/reused samples; metadata now has {len(merged_samples)} samples and {len(merged_failures)} failures")


if __name__ == "__main__":
    main()
