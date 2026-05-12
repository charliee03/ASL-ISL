import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.recognition.preprocess import HandKeypointExtractor


class BatchFeatureExtractor:
    def __init__(self, num_frames=16, sample_strategy="uniform"):
        self.num_frames = num_frames
        self.sample_strategy = sample_strategy
        self.extractor = HandKeypointExtractor()

    def extract_from_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        if self.sample_strategy == "uniform":
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            step = max(1, total // self.num_frames)
            indices = np.arange(0, total, step)[:self.num_frames]
        keypoints = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                keypoints.append(np.zeros((21, 3), dtype=np.float32))
                continue
            kp = self.extractor.extract(frame)
            if len(kp) == 0:
                kp = np.zeros((21, 3), dtype=np.float32)
            keypoints.append(kp)
        cap.release()
        return np.stack(keypoints)

    def process_dataset(self, video_dir, annotation_file, output_dir):
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(annotation_file) as f:
            annotations = json.load(f)
        metadata = []
        for ann in tqdm(annotations, desc="Extracting features"):
            video_path = video_dir / ann["video"]
            if not video_path.exists():
                continue
            features = self.extract_from_video(str(video_path))
            if features is None:
                continue
            out_path = output_dir / f"{Path(ann['video']).stem}.npy"
            np.save(out_path, features)
            metadata.append({
                "video": ann["video"],
                "features": str(out_path),
                "gloss": ann.get("gloss", ""),
                "split": ann.get("split", "")
            })
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {len(metadata)} feature files to {output_dir}")
        return metadata


def extract_keypoints_from_frame(frame, extractor=None):
    if extractor is None:
        extractor = HandKeypointExtractor()
    kp = extractor.extract(frame)
    if len(kp) == 0:
        return np.zeros((21, 3), dtype=np.float32)
    return kp
