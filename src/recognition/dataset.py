import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# pyrefly: ignore [missing-import]
from src.recognition.preprocess import HandKeypointExtractor


class RandomRotation:
    def __init__(self, degree_range=13):
        self.degree_range = degree_range

    def __call__(self, sample):
        keypoints = sample["keypoints"].copy()
        angle = np.random.uniform(-self.degree_range, self.degree_range)
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        # Rotate x and y coordinates around origin (nose)
        x = keypoints[:, :, 0]
        y = keypoints[:, :, 1]
        
        keypoints[:, :, 0] = x * cos_a - y * sin_a
        keypoints[:, :, 1] = x * sin_a + y * cos_a
        
        sample["keypoints"] = keypoints
        return sample


class RandomSqueeze:
    def __init__(self, squeeze_ratio=0.15):
        self.squeeze_ratio = squeeze_ratio

    def __call__(self, sample):
        keypoints = sample["keypoints"].copy()
        factor_x = np.random.uniform(1.0 - self.squeeze_ratio, 1.0 + self.squeeze_ratio)
        factor_y = np.random.uniform(1.0 - self.squeeze_ratio, 1.0 + self.squeeze_ratio)
        
        keypoints[:, :, 0] = keypoints[:, :, 0] * factor_x
        keypoints[:, :, 1] = keypoints[:, :, 1] * factor_y
        
        sample["keypoints"] = keypoints
        return sample


class RandomMirror:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.p:
            keypoints = sample["keypoints"].copy()
            # Flip X coordinate
            keypoints[:, :, 0] = -keypoints[:, :, 0]
            
            # Swap left hand (0 to 9) and right hand (10 to 19) keypoints
            left_hand = keypoints[:, 0:10].copy()
            right_hand = keypoints[:, 10:20].copy()
            keypoints[:, 0:10] = right_hand
            keypoints[:, 10:20] = left_hand
            
            # Swap pose components (shoulders 20 <-> 21, elbows 22 <-> 23, wrists 24 <-> 25)
            pose = keypoints[:, 20:27].copy()
            pose_swapped = pose.copy()
            pose_swapped[:, 0] = pose[:, 1]
            pose_swapped[:, 1] = pose[:, 0]
            pose_swapped[:, 2] = pose[:, 3]
            pose_swapped[:, 3] = pose[:, 2]
            pose_swapped[:, 4] = pose[:, 5]
            pose_swapped[:, 5] = pose[:, 4]
            
            keypoints[:, 20:27] = pose_swapped
            sample["keypoints"] = keypoints
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class WLASLDataset(Dataset):
    def __init__(self, data_root, annotation_file, split="train",
                 num_frames=16, transform=None, limit=None):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.transform = transform
        self.extractor = HandKeypointExtractor()
        
        # Setup cache directory
        self.cache_dir = self.data_root / "cached_keypoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples = []
        self.classes = set()
        
        with open(annotation_file) as f:
            data = json.load(f)
            
        # Detect if the annotations structure is nested list, dict, or flat list
        if isinstance(data, list) and len(data) > 0 and "instances" in data[0]:
            # 1. Standard WLASL nested structure
            for entry in data:
                gloss = entry["gloss"]
                self.classes.add(gloss)
                for inst in entry["instances"]:
                    if inst.get("split") == split:
                        self.samples.append({
                            "video": f"videos/{inst['video_id']}.mp4",
                            "gloss": gloss,
                            "split": split
                        })
            self.classes = sorted(list(self.classes))
        elif isinstance(data, dict):
            # 2. NSLT dictionary format (e.g., nslt_100.json, nslt_300.json)
            class_list_file = self.data_root / "wlasl_class_list.txt"
            class_id_to_gloss = {}
            if class_list_file.exists():
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
                                
            # First gather the classes for all samples in the entire dataset
            for video_id, info in data.items():
                action = info.get("action", [])
                if len(action) > 0:
                    class_id = action[0]
                    gloss = class_id_to_gloss.get(class_id, f"class_{class_id}")
                    self.classes.add(gloss)
            self.classes = sorted(list(self.classes))
            
            # Now filter the samples by split
            for video_id, info in data.items():
                split_name = info.get("subset")
                if split_name == split:
                    action = info.get("action", [])
                    if len(action) > 0:
                        class_id = action[0]
                        gloss = class_id_to_gloss.get(class_id, f"class_{class_id}")
                        self.samples.append({
                            "video": f"videos/{video_id}.mp4",
                            "gloss": gloss,
                            "split": split
                        })
        else:
            # 3. Flattened structure (or mock annotations)
            for ann in data:
                self.classes.add(ann["gloss"])
            self.classes = sorted(list(self.classes))
            
            for ann in data:
                if ann.get("split") == split:
                    self.samples.append(ann)
            
        if limit is not None:
            self.samples = self.samples[:limit]
            
        self.word2idx = {w: i for i, w in enumerate(self.classes)}
        
        # Count and print cached files for this configuration
        cached_files = list(self.cache_dir.glob(f"*_f{self.num_frames}.npy"))
        print(f"[Dataset] Split '{split}': loaded {len(self.samples)} samples. Cache status: {len(cached_files)} keypoint sequence(s) cached in '{self.cache_dir}'.")

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path):
        video_name = Path(video_path).stem
        cache_file = self.cache_dir / f"{video_name}_f{self.num_frames}.npy"
        
        # Load from cache if it exists
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                # If cached file is corrupted, fall back to extracting again
                pass

        if not video_path.exists():
            return np.zeros((self.num_frames, 27, 3), dtype=np.float32)

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return np.zeros((self.num_frames, 27, 3), dtype=np.float32)
            
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        keypoints = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                keypoints.append(np.zeros((27, 3), dtype=np.float32))
                continue
            kp = self.extractor.extract(frame)
            keypoints.append(kp)
        cap.release()
        
        stacked = np.stack(keypoints)
        
        # Save to cache for next time
        try:
            np.save(cache_file, stacked)
        except Exception as e:
            print(f"Warning: could not save cache for {video_name}: {e}")
            
        return stacked

    def __getitem__(self, idx):
        ann = self.samples[idx]
        video_path = self.data_root / ann["video"]
        keypoints = self._sample_frames(video_path)
        label = self.word2idx[ann["gloss"]]
        
        sample = {
            "keypoints": keypoints.astype(np.float32), 
            "label": label, 
            "gloss": ann["gloss"]
        }
        if self.transform:
            sample = self.transform(sample)
            
        sample["keypoints"] = torch.FloatTensor(sample["keypoints"])
        return sample


class ISLDataset(Dataset):
    def __init__(self, features_dir, annotation_file, split="train", transform=None):
        self.features_dir = Path(features_dir)
        self.transform = transform

        with open(annotation_file) as f:
            annotations = json.load(f)
        self.samples = [ann for ann in annotations if ann.get("split") == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        npy_path = self.features_dir / ann["features"]
        
        # Load pre-extracted keypoints
        keypoints = np.load(npy_path).astype(np.float32)
        
        sample = {
            "keypoints": keypoints,
            "asl_gloss": ann.get("asl_gloss", ""),
            "isl_gloss": ann.get("isl_gloss", "")
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        sample["keypoints"] = torch.FloatTensor(sample["keypoints"])
        return sample


def collate_keypoints(batch):
    keypoints = torch.stack([b["keypoints"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    glosses = [b["gloss"] for b in batch]
    return {"keypoints": keypoints, "label": labels, "gloss": glosses}
