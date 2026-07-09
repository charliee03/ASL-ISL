#!/usr/bin/env python3
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from tqdm import tqdm

from src.recognition.preprocess import HandKeypointExtractor

def process_video(args):
    video_path, cache_file, num_frames = args
    video_path = Path(video_path)
    cache_file = Path(cache_file)
    
    if cache_file.exists():
        return True
        
    if not video_path.exists():
        return False

    extractor = HandKeypointExtractor()
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        cap.release()
        return False
        
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    keypoints = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            keypoints.append(np.zeros((27, 3), dtype=np.float32))
            continue
        kp = extractor.extract(frame)
        keypoints.append(kp)
        
    cap.release()
    stacked = np.stack(keypoints)
    
    try:
        np.save(cache_file, stacked)
        return True
    except Exception:
        return False


def build_cache(data_root="data/msasl/MS-ASL", annotation_file="data/msasl/MS-ASL/MSASL_unified.json", num_frames=32):
    data_root = Path(data_root)
    cache_dir = data_root / "cached_keypoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file) as f:
        data = json.load(f)
        
    tasks = []
    for entry in data:
        video_path = data_root / entry["video"]
        video_name = video_path.stem
        cache_file = cache_dir / f"{video_name}_f{num_frames}.npy"
        
        if not cache_file.exists() and video_path.exists():
            tasks.append((str(video_path), str(cache_file), num_frames))
            
    if not tasks:
        print("All videos are already cached! You are ready to train.")
        return
        
    num_cores = max(1, cpu_count() - 2)
    print(f"Found {len(tasks)} videos to extract keypoints for.")
    print(f"Starting parallel extraction using {num_cores} CPU cores... (This will maximize CPU usage)")
    
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_video, tasks), total=len(tasks), desc="Extracting Keypoints"))
        
    success = sum(results)
    print(f"\nFinished! Successfully cached {success}/{len(tasks)} videos.")
    print("You can now run train_recognition.py and it will start instantly!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/msasl/MS-ASL")
    parser.add_argument("--annotation-file", default="data/msasl/MS-ASL/MSASL_unified.json")
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()
    
    build_cache(args.data_root, args.annotation_file, args.num_frames)
