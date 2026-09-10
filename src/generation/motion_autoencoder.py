"""Small temporal motion autoencoder used as a pose-generation prerequisite.

It reconstructs an observed landmark sequence from a fixed-size latent vector.
It is intentionally not text-conditioned and must not be described as a sign
generator.  Its purpose is to establish that the extracted motion corpus can be
modelled and evaluated before attempting conditional generation.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalMotionAutoencoder(nn.Module):
    """GRU sequence autoencoder for `[batch, frames, keypoints, xyz]` motion."""

    def __init__(self, num_keypoints: int = 75, hidden_dim: int = 192, latent_dim: int = 96):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.feature_dim = num_keypoints * 3
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.to_latent = nn.Sequential(nn.Linear(hidden_dim * 2, latent_dim), nn.Tanh())
        self.decoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.frame_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.feature_dim)
        )

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if motion.ndim != 4 or motion.shape[-2:] != (self.num_keypoints, 3):
            raise ValueError(
                f"Expected [batch, frames, {self.num_keypoints}, 3], got {tuple(motion.shape)}"
            )
        batch_size, frames = motion.shape[:2]
        encoded, hidden = self.encoder(self.frame_encoder(motion.flatten(2)))
        # Bidirectional GRU final state; using it rather than a temporal mean keeps
        # the latent representation bounded and makes sequence order meaningful.
        latent = self.to_latent(torch.cat((hidden[-2], hidden[-1]), dim=-1))
        decoded, _ = self.decoder(latent.unsqueeze(1).expand(batch_size, frames, -1))
        reconstruction = self.frame_decoder(decoded).view(batch_size, frames, self.num_keypoints, 3)
        return reconstruction, latent
