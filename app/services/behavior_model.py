"""Runtime loader and predictor for dcad_behavior_model checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.services.behavior_labels import BEHAVIOR_LABELS
from app.services.pipeline import FrameFeatures


class BehaviorClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BehaviorPrediction:
    label: str
    confidence: float


class BehaviorModelRuntime:
    def __init__(self, model: BehaviorClassifier, labels: list[str], device: torch.device):
        self.model = model
        self.labels = labels
        self.device = device

    @property
    def available(self) -> bool:
        return True

    @staticmethod
    def _feature_vector(features: list[FrameFeatures]) -> np.ndarray:
        if not features:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        mean_intensity = np.array([f.mean_intensity for f in features], dtype=np.float32)
        motion = np.array([f.motion_score for f in features], dtype=np.float32)
        density_hint = float(np.clip(motion.mean() / 6.0, 0.0, 1.0))
        return np.array(
            [
                float(mean_intensity.mean()),
                float(mean_intensity.std()),
                float(motion.mean()),
                float(motion.std()),
                density_hint,
                float(len(features)),
            ],
            dtype=np.float32,
        )

    def predict(self, features: list[FrameFeatures]) -> BehaviorPrediction:
        vector = self._feature_vector(features)
        x = torch.from_numpy(vector).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            score, idx = torch.max(probs, dim=1)

        label_idx = int(idx.item())
        if label_idx < 0 or label_idx >= len(self.labels):
            return BehaviorPrediction(label="normal_flow", confidence=float(score.item()))
        return BehaviorPrediction(label=self.labels[label_idx], confidence=float(score.item()))


class BehaviorModelFallback:
    available = False

    def predict(self, features: list[FrameFeatures]) -> BehaviorPrediction:  # pragma: no cover
        return BehaviorPrediction(label="normal_flow", confidence=0.0)


def load_behavior_model(checkpoint_path: Path) -> BehaviorModelRuntime | BehaviorModelFallback:
    if not checkpoint_path.exists():
        print(f"[WARN] Behavior checkpoint not found: {checkpoint_path}")
        return BehaviorModelFallback()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        payload = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Cannot load behavior checkpoint: {exc}")
        return BehaviorModelFallback()

    state_dict = payload.get("state_dict") if isinstance(payload, dict) else payload
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    labels = meta.get("labels", BEHAVIOR_LABELS)

    model = BehaviorClassifier(input_dim=6, num_classes=len(labels)).to(device)
    try:
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Invalid checkpoint state dict: {exc}")
        return BehaviorModelFallback()

    print(f"[INFO] Loaded behavior model from: {checkpoint_path}")
    return BehaviorModelRuntime(model=model, labels=labels, device=device)
