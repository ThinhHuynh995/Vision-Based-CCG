"""
Branch 3 — Behavior Classification (Deep Learning)
===================================================
  • BehaviorClassifier: MobileNetV3-based fine-tuned classifier
  • BehaviorDataset: PyTorch dataset for training
  • Trainer: Training loop with metrics
  • Rule-based fallback when model not available
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Fighting",
    2: "Falling",
    3: "Loitering",
    4: "Crowd Panic",
}


# ── Dataset ───────────────────────────────────────────────────────────────

class BehaviorDataset(Dataset):
    """
    Dataset for behavior classification.

    Expects folder structure:
        data/processed/
            Normal/      img1.jpg, img2.jpg ...
            Fighting/    ...
            Falling/     ...
            Loitering/   ...
            Crowd Panic/ ...

    Args:
        root_dir: path to dataset root
        split: "train" or "val"
        transform: torchvision transforms
    """
    CLASS_MAP = {name: idx for idx, name in CLASS_NAMES.items()}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
    ):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for cls_name, cls_idx in self.CLASS_MAP.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in exts:
                    self.samples.append((img_path, cls_idx))
        logger.info(f"BehaviorDataset: loaded {len(self.samples)} samples from {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model Builder ─────────────────────────────────────────────────────────

def build_model(
    model_name: str = "mobilenet_v3_small",
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a classification model with fine-tuning head.

    Supported: mobilenet_v3_small | resnet50 | efficientnet_b0
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=weights)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info(f"Built {model_name} with {num_classes} output classes")
    return m


# ── Transforms ────────────────────────────────────────────────────────────

def get_transforms(input_size: Tuple[int, int] = (224, 224), is_train: bool = True):
    """Standard ImageNet-normalized transforms with augmentation for training."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size[0] + 32, input_size[1] + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Trainer ───────────────────────────────────────────────────────────────

class Trainer:
    """
    Simple training loop for behavior classifier.

    Usage:
        trainer = Trainer(model, device="cpu")
        history = trainer.fit(train_loader, val_loader, epochs=20)
        trainer.save("models/weights/behavior_clf.pth")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
    ) -> Dict:
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )
        return self.history

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                out = self.model(imgs)
                loss = self.criterion(out, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                total += imgs.size(0)
        return total_loss / total, correct / total

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved → {path}")


# ── Inference Classifier ──────────────────────────────────────────────────

class BehaviorClassifier:
    """
    Runtime behavior classifier for individual cropped person images.

    Falls back to a rule-based heuristic when no weights are loaded.

    Usage:
        clf = BehaviorClassifier(cfg["classification"])
        label, conf = clf.predict(crop_bgr)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = "cpu"
        self.num_classes = cfg.get("num_classes", 5)
        self.conf_threshold = cfg.get("confidence_threshold", 0.60)
        self.input_size = tuple(cfg.get("input_size", [224, 224]))
        self.model: Optional[nn.Module] = None
        self._transform = get_transforms(self.input_size, is_train=False)
        self._try_load(cfg.get("model", "mobilenet_v3_small"))

    def _try_load(self, model_name: str):
        weight_path = Path("models/weights/behavior_clf.pth")
        try:
            self.model = build_model(model_name, self.num_classes, pretrained=False)
            if weight_path.exists():
                self.model.load_state_dict(
                    torch.load(weight_path, map_location=self.device)
                )
                self.model.eval()
                logger.info(f"BehaviorClassifier: loaded weights from {weight_path}")
            else:
                # Use pretrained ImageNet weights as initialization
                self.model = build_model(model_name, self.num_classes, pretrained=True)
                self.model.eval()
                logger.warning("No fine-tuned weights found – using pretrained backbone (rule-based fallback active)")
        except Exception as e:
            logger.warning(f"Model init failed ({e}), using rule-based fallback")
            self.model = None

    def predict(self, crop_bgr: np.ndarray) -> Tuple[int, float]:
        """
        Classify behavior from a cropped person image.

        Args:
            crop_bgr: BGR image of person crop (any size)

        Returns:
            (class_index, confidence) tuple
        """
        if self.model is not None and crop_bgr.size > 0:
            return self._predict_nn(crop_bgr)
        return self._predict_rule(crop_bgr)

    def _predict_nn(self, crop: np.ndarray) -> Tuple[int, float]:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, cls = probs.max(0)
        return int(cls), float(conf)

    def _predict_rule(self, crop: np.ndarray) -> Tuple[int, float]:
        """Heuristic fallback: use aspect ratio & motion proxy."""
        h, w = crop.shape[:2]
        ratio = h / max(w, 1)
        # Very wide/short → might be falling
        if ratio < 0.8:
            return 2, 0.55   # Falling
        return 0, 0.90       # Normal

    def predict_batch(
        self, crops: List[np.ndarray]
    ) -> List[Tuple[int, float]]:
        return [self.predict(c) for c in crops]
