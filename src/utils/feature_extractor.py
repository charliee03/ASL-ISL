import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# pyrefly: ignore [missing-import]
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


class ISLFeatureExtractor:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames
        self.extractor = HandKeypointExtractor()

    def extract_robust_keypoints(self, frame):
        kp = self.extractor.extract(frame)
        # Handle MediaPipe list output and guarantee (21, 3) shape
        if len(kp) == 0:
            return np.zeros((21, 3), dtype=np.float32)
        kp_arr = np.array(kp, dtype=np.float32)
        if kp_arr.shape[0] >= 63:
            return kp_arr[:63].reshape((21, 3))
        return np.zeros((21, 3), dtype=np.float32)

    def extract_from_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        # Uniform sampling indices
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        keypoints = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                keypoints.append(np.zeros((21, 3), dtype=np.float32))
                continue
            kp = self.extract_robust_keypoints(frame)
            keypoints.append(kp)
        cap.release()
        return np.stack(keypoints)

    def process_dataset(self, video_dir, gloss_csv_path, output_dir):
        video_dir = Path(video_dir)
        gloss_csv_path = Path(gloss_csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not gloss_csv_path.exists():
            raise FileNotFoundError(f"Gloss CSV not found at {gloss_csv_path}")

        # Load annotation CSV
        df = pd.read_csv(gloss_csv_path)
        df.columns = ["sentence", "isl_gloss"]
        df = df.dropna().reset_index(drop=True)
        # Create mapping of normalized sentence key (all spaces removed, lowercase) to (original sentence, single-spaced gloss)
        sentence_to_gloss = {}
        for _, row in df.iterrows():
            sent = str(row["sentence"]).strip()
            # Normalize whitespace inside gloss
            gloss = " ".join(str(row["isl_gloss"]).strip().split())
            key = "".join(sent.lower().split())
            sentence_to_gloss[key] = (sent, gloss)

        # Scan video_dir for directories of sentence categories
        sentence_dirs = [d for d in video_dir.iterdir() if d.is_dir()]
        
        # Collect all video paths grouped by sentence category to do a stratified split
        videos_by_sentence = {}
        for s_dir in sentence_dirs:
            sentence_key = "".join(s_dir.name.lower().split())
            if sentence_key not in sentence_to_gloss:
                continue
            video_files = list(s_dir.glob("*.mp4")) + list(s_dir.glob("*.MP4")) + list(s_dir.glob("*.avi"))
            if len(video_files) > 0:
                videos_by_sentence[s_dir.name] = video_files

        # Set fixed random seed for deterministic stratified split
        random.seed(42)

        metadata = []
        # Process sentence by sentence and divide samples stratified by category
        for sentence_name, video_files in tqdm(videos_by_sentence.items(), desc="Processing ISL Dataset Splits"):
            sentence_key = "".join(sentence_name.lower().split())
            _, isl_gloss = sentence_to_gloss[sentence_key]
            
            # Shuffle videos deterministically after sorting them for absolute cross-platform reproducibility
            shuffled_videos = sorted(video_files)
            random.shuffle(shuffled_videos)

            n_videos = len(shuffled_videos)
            train_idx = int(0.70 * n_videos)
            val_idx = train_idx + int(0.15 * n_videos)
            # Ensure at least 1 video is in train/val/test if possible
            if train_idx == 0 and n_videos > 0:
                train_idx = 1
            if val_idx == train_idx and n_videos > train_idx:
                val_idx = train_idx + 1

            for i, v_path in enumerate(shuffled_videos):
                # Determine stratified split
                if i < train_idx:
                    split = "train"
                elif i < val_idx:
                    split = "val"
                else:
                    split = "test"

                # Save features as unique name to prevent collisions
                safe_sentence = sentence_name.replace(" ", "_").replace("(", "").replace(")", "")
                safe_name = v_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
                out_filename = f"{safe_sentence}_{safe_name}.npy"
                out_path = output_dir / out_filename

                # Check if features are already extracted to avoid slow MediaPipe run
                features = None
                if out_path.exists():
                    try:
                        features = np.load(out_path)
                    except Exception:
                        features = None

                if features is None:
                    # Extract video features
                    features = self.extract_from_video(v_path)
                    if features is None:
                        continue
                    np.save(out_path, features)

                metadata.append({
                    "video": str(v_path.relative_to(video_dir.parent)),
                    "features": out_filename,
                    "asl_gloss": "", # Optional, maps to ASL counterparts if trained
                    "isl_gloss": isl_gloss,
                    "split": split
                })

        # Save metadata.json
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Successfully processed {len(metadata)} video files with 70/15/15 stratified split.")
        print(f"Features and metadata.json saved to {output_dir}")
        return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AITE Feature Extractor")
    parser.add_argument("--mode", choices=["wlasl", "isl"], default="isl", help="Dataset mode: 'isl' (ISLFeatureExtractor) or 'wlasl' (BatchFeatureExtractor)")
    parser.add_argument("--video_dir", required=True, help="Path to the video directory")
    parser.add_argument("--annotation_file", required=True, help="Path to the annotations")
    parser.add_argument("--output_dir", required=True, help="Path to save extracted features")

    args = parser.parse_args()

    if args.mode == "isl":
        print("Initializing ISL Feature Extractor...")
        extractor = ISLFeatureExtractor()
        extractor.process_dataset(args.video_dir, args.annotation_file, args.output_dir)
    else:
        print("Initializing WLASL Batch Feature Extractor...")
        extractor = BatchFeatureExtractor()
        extractor.process_dataset(args.video_dir, args.annotation_file, args.output_dir)

