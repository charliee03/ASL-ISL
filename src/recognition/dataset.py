import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# pyrefly: ignore [missing-import]
from src.recognition.preprocess import HandKeypointExtractor


class WLASLDataset(Dataset):
    def __init__(self, data_root, annotation_file, split="train",
                 num_frames=16, transform=None):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.transform = transform
        self.extractor = HandKeypointExtractor()
        with open(annotation_file) as f:
            annotations = json.load(f)
        self.samples = [ann for ann in annotations if ann.get("split") == split]
        self.classes = sorted({ann["gloss"] for ann in annotations})
        self.word2idx = {w: i for i, w in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return np.zeros((self.num_frames, 21, 3), dtype=np.float32)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        keypoints = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                keypoints.append(np.zeros((21, 3), dtype=np.float32))
                continue
            kp = self.extractor.extract(frame)
            keypoints.append(kp)
        cap.release()
        return np.stack(keypoints)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        video_path = self.data_root / ann["video"]
        keypoints = self._sample_frames(video_path)
        label = self.word2idx[ann["gloss"]]
        sample = {"keypoints": torch.FloatTensor(keypoints), "label": label, "gloss": ann["gloss"]}
        if self.transform:
            sample = self.transform(sample)
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
        
        # Load pre-extracted keypoints, shape (16, 21, 3)
        keypoints = np.load(npy_path).astype(np.float32)
        
        sample = {
            "keypoints": keypoints,
            "asl_gloss": ann.get("asl_gloss", ""),
            "isl_gloss": ann.get("isl_gloss", "")
        }
        
        # Apply augmentation on numpy array
        if self.transform:
            sample = self.transform(sample)
            
        # Convert to tensor AFTER augmentation
        sample["keypoints"] = torch.FloatTensor(sample["keypoints"])
        
        return sample


def collate_keypoints(batch):
    keypoints = torch.stack([b["keypoints"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    glosses = [b["gloss"] for b in batch]
    return {"keypoints": keypoints, "label": labels, "gloss": glosses}
