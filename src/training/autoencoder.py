from __future__ import annotations

import numpy as np


class PCAAutoencoder:
    def __init__(self):
        self.mean = None
        self.std = None
        self.components = None
        self.explained_variance = None
        self.total_variance_explained = 0.0

    def _randomized_svd(self, X, k, n_iter=3):
        n, d = X.shape
        omega = np.random.randn(d, k + 10).astype(np.float32)
        Y = X @ omega
        for _ in range(n_iter):
            Y = X @ (X.T @ Y)
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ X
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat
        return U[:, :k], S[:k], Vt[:k, :]

    def fit(self, X_normal: np.ndarray, n_components: int = 32):
        self.mean = X_normal.mean(axis=0)
        self.std = X_normal.std(axis=0) + 1e-8
        X = (X_normal - self.mean) / self.std
        if X.shape[0] > 5000:
            _, S, Vt = self._randomized_svd(X, n_components)
        else:
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
            S, Vt = S[:n_components], Vt[:n_components]
        self.components = Vt
        self.explained_variance = (S**2) / max(X.shape[0] - 1, 1)
        self.total_variance_explained = float(self.explained_variance.sum() / (((X**2).sum() / max(X.shape[0] - 1, 1)) + 1e-8))

    def reconstruct_error(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / self.std
        proj = Xn @ self.components.T
        rec = proj @ self.components
        return ((Xn - rec) ** 2).mean(axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        err = self.reconstruct_error(X)
        p5, p95 = np.percentile(err, [5, 95])
        return np.clip((err - p5) / (p95 - p5 + 1e-8), 0, 1)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, components=self.components, explained_variance=self.explained_variance, total_variance_explained=np.array([self.total_variance_explained]))

    def load(self, path):
        d = np.load(path)
        self.mean = d["mean"]
        self.std = d["std"]
        self.components = d["components"]
        self.explained_variance = d["explained_variance"]
        self.total_variance_explained = float(d["total_variance_explained"])
