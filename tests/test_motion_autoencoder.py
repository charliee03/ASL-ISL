import torch

from src.generation.motion_autoencoder import TemporalMotionAutoencoder


def test_motion_autoencoder_preserves_motion_shape_and_latent_size():
    model = TemporalMotionAutoencoder(num_keypoints=75, hidden_dim=32, latent_dim=12)
    reconstruction, latent = model(torch.zeros(2, 9, 75, 3))
    assert reconstruction.shape == (2, 9, 75, 3)
    assert latent.shape == (2, 12)
