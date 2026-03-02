"""Mô-đun mô phỏng DCAD với CSRNet backbone và Crowd Context Gate."""

from dataclasses import dataclass

import torch
import torch.nn as nn


class TinyCSRNetHead(nn.Module):
    """Khối đại diện cho output density map từ CSRNet pretrained."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.density_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.density_head(x)


@dataclass
class DCADOutput:
    anomaly_score: torch.Tensor
    density_score: torch.Tensor
    threshold: torch.Tensor


class CrowdContextGate(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, density_score: torch.Tensor, base_threshold: float = 0.5) -> torch.Tensor:
        density_score = density_score.view(-1, 1)
        alpha = self.gate(density_score)
        return base_threshold + 0.3 * alpha


class DCADModel(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.anomaly_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.ccg = CrowdContextGate()

    def forward(self, video_feat: torch.Tensor, density_score: torch.Tensor) -> DCADOutput:
        anomaly_score = self.anomaly_head(video_feat)
        threshold = self.ccg(density_score)
        return DCADOutput(anomaly_score=anomaly_score, density_score=density_score, threshold=threshold)
