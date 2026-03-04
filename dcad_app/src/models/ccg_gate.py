from __future__ import annotations

import numpy as np


class CCGGate:
    def __init__(self, tau_base: float = 0.5, alpha: float = 0.3):
        self.gate_values = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        self.tau_base = tau_base
        self.alpha = alpha

    def calibrate(self, scores: np.ndarray, zones: np.ndarray, labels: np.ndarray, epochs: int = 30, lr: float = 0.01):
        for epoch in range(epochs):
            total_loss = 0.0
            for z in range(3):
                mask = zones == z
                if mask.sum() == 0:
                    continue
                s_z = scores[mask]
                y_z = labels[mask].astype(np.float32)
                tau_z = self.threshold(z)
                loss_normal = ((s_z - tau_z) ** 2 * (1 - y_z)).mean()
                diff_anom = np.maximum(0.0, tau_z - s_z)
                loss_anom = (diff_anom**2 * y_z).mean()
                loss_z = float(loss_normal + loss_anom)
                d1 = (2 * (tau_z - s_z) * (1 - y_z)).mean()
                d2 = (2 * diff_anom * y_z).mean()
                dL_dtau = float(d1 + d2)
                self.gate_values[z] = np.clip(self.gate_values[z] - lr * dL_dtau * self.alpha, 0.0, 1.0)
                total_loss += loss_z
            yield {
                "epoch": epoch + 1,
                "loss": float(total_loss),
                "gate_low": float(self.gate_values[0]),
                "gate_mid": float(self.gate_values[1]),
                "gate_high": float(self.gate_values[2]),
            }

    def threshold(self, zone_idx: int) -> float:
        return float(self.tau_base + self.alpha * self.gate_values[zone_idx])

    def save(self, path):
        np.savez(path, gate_values=self.gate_values, tau_base=np.array([self.tau_base]), alpha=np.array([self.alpha]))

    def load(self, path):
        d = np.load(path)
        self.gate_values = d["gate_values"]
        self.tau_base = float(d["tau_base"])
        self.alpha = float(d["alpha"])
