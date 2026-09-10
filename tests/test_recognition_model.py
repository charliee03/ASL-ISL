import torch
import numpy as np

from src.recognition.dataset import RandomTemporalCrop
from src.recognition.model import SignRecognitionTransformer


def test_motion_aware_transformer_accepts_pose_hand_sequences():
    model = SignRecognitionTransformer(
        num_keypoints=75,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        vocab_size=100,
        dropout=0.0,
        use_velocity=True,
        use_presence=True,
    )
    output = model(torch.zeros(2, 32, 75, 3))
    assert output.shape == (2, 100)


def test_temporal_crop_preserves_sequence_shape_and_missing_landmarks():
    sequence = np.zeros((32, 75, 3), dtype=np.float32)
    sequence[:, :33] = 1.0
    result = RandomTemporalCrop(0.9)({"keypoints": sequence})["keypoints"]
    assert result.shape == sequence.shape
    assert np.all(result[:, 33:] == 0.0)
