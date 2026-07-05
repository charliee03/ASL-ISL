import numpy as np

class HorizontalMirror:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        u = np.random.uniform(0, 1)
        if u >= self.p:
            return sample
        
        # In-place operation on a copy or modify directly (we'll modify directly, but best to copy keypoints)
        keypoints = sample["keypoints"].copy()
        
        # Flip the x-coordinate (index 0)
        keypoints[:, :, 0] = 1.0 - keypoints[:, :, 0]
        
        sample["keypoints"] = keypoints
        return sample


class TemporalJitter:
    def __init__(self, max_jitter=2):
        self.max_jitter = max_jitter

    def __call__(self, sample):
        keypoints = sample["keypoints"].copy()
        T = keypoints.shape[0]
        
        n_drop = np.random.randint(0, self.max_jitter + 1)
        n_dup = np.random.randint(0, self.max_jitter + 1)
        
        if n_drop > 0:
            drop_indices = np.random.choice(T, n_drop, replace=False)
            keypoints = np.delete(keypoints, drop_indices, axis=0)
            
        if n_dup > 0:
            dup_indices = np.random.choice(keypoints.shape[0], n_dup, replace=True)
            # np.insert inserts before the given index, duplicate the frame
            keypoints = np.insert(keypoints, dup_indices, keypoints[dup_indices], axis=0)
            
        # Resample back to original length T
        resampled_indices = np.linspace(0, keypoints.shape[0] - 1, T, dtype=int)
        sample["keypoints"] = keypoints[resampled_indices]
        return sample


class GaussianNoise:
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, sample):
        keypoints = sample["keypoints"].copy()
        
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=keypoints.shape).astype(np.float32)
        keypoints = keypoints + noise
        
        # Clip to [0.0, 1.0] range
        keypoints = np.clip(keypoints, 0.0, 1.0)
        
        sample["keypoints"] = keypoints
        return sample


class ScaleAugmentation:
    def __init__(self, scale_range=0.10):
        self.scale_range = scale_range

    def __call__(self, sample):
        keypoints = sample["keypoints"].copy()
        
        scale = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
        
        # Compute centroid per frame (mean across the 21 keypoints)
        centroid = keypoints.mean(axis=1, keepdims=True)
        
        # Scale around centroid
        keypoints = centroid + scale * (keypoints - centroid)
        
        # Clip to valid range
        keypoints = np.clip(keypoints, 0.0, 1.0)
        
        sample["keypoints"] = keypoints
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentation_pipeline(config):
    transforms = []
    
    if config.get("mirror_prob", 0) > 0:
        transforms.append(HorizontalMirror(p=config["mirror_prob"]))
        
    if config.get("max_jitter", 0) > 0:
        transforms.append(TemporalJitter(max_jitter=config["max_jitter"]))
        
    if config.get("noise_sigma", 0) > 0:
        transforms.append(GaussianNoise(sigma=config["noise_sigma"]))
        
    if config.get("scale_range", 0) > 0:
        transforms.append(ScaleAugmentation(scale_range=config["scale_range"]))
        
    return Compose(transforms)
