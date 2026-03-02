"""Script huấn luyện mẫu cho DCAD."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.services.dcad_model import DCADModel


def run_demo_training(epochs: int = 3) -> None:
    x = torch.rand(256, 128)
    density = torch.rand(256, 1)
    y = (torch.rand(256, 1) > 0.7).float()

    dataset = TensorDataset(x, density, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DCADModel(feature_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total = 0.0
        for feat, d, label in loader:
            output = model(feat, d)
            adjusted = torch.clamp(output.anomaly_score - output.threshold + 0.5, 0.0, 1.0)
            loss = loss_fn(adjusted, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - loss: {total / len(loader):.4f}")


if __name__ == "__main__":
    run_demo_training()
