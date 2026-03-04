"""
src/classification/classifier.py — Branch 3: Behavior Classification
=====================================================================
  • build_model()      – MobileNetV3 / EfficientNet-B0 / ResNet-50
  • get_transforms()   – ImageNet-normalized transforms
  • BehaviorDataset    – Dataset từ data/processed/{train|val|test}/<class>/
  • Trainer            – Training loop với early stopping
  • BehaviorClassifier – Runtime inference (fallback rule-based nếu chưa có weights)
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import src.utils.log as _log

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    Dataset = object  # placeholder

log = _log.get(__name__)

CLASS_NAMES: Dict[int, str] = {
    0: "Normal", 1: "Fighting", 2: "Falling", 3: "Loitering", 4: "Crowd Panic",
}
CLASS_IDX = {v: k for k, v in CLASS_NAMES.items()}


# ── Dataset ──────────────────────────────────────────────────────────────────

class BehaviorDataset(Dataset):
    """
    Cấu trúc thư mục:
        data/processed/{split}/{ClassName}/*.jpg

    Sử dụng:
        ds = BehaviorDataset("data/processed", split="train", transform=tf)
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.transform = transform
        self._samples: List[Tuple[Path, int]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for cls_name, cls_id in CLASS_IDX.items():
            cls_dir = Path(root) / split / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in exts:
                    self._samples.append((p, cls_id))

        log.info(f"BehaviorDataset [{split}]: {len(self._samples)} ảnh, "
                 f"{len(set(s[1] for s in self._samples))} lớp")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(img)
        if self.transform:
            pil = self.transform(pil)
        return pil, label


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(size: Tuple[int,int] = (224,224), train: bool = True):
    if not _TORCH_OK:
        return None
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((size[0]+28, size[1]+28)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(12),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_model(name: str = "mobilenet_v3_small", n_classes: int = 5,
                pretrained: bool = True):
    if not _TORCH_OK:
        raise ImportError("torch/torchvision chưa được cài đặt")
    w = "DEFAULT" if pretrained else None
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(f"Model không hỗ trợ: {name}")
    log.info(f"Model: {name} | {n_classes} classes | pretrained={pretrained}")
    return m


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Training loop với early stopping và LR scheduler.

    Sử dụng:
        trainer = Trainer(model, device="cpu")
        history = trainer.fit(train_dl, val_dl, epochs=25)
        trainer.save("models/weights/clf.pth")
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 5e-4, weight_decay: float = 1e-4):
        self.model    = model.to(device)
        self.device   = device
        self.crit     = nn.CrossEntropyLoss()
        self.opt      = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched    = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=25)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

    def fit(self, train_dl: DataLoader, val_dl: DataLoader,
            epochs: int = 25, patience: int = 7) -> Dict:
        best_val, wait, best_state = 0.0, 0, None

        for ep in range(1, epochs+1):
            tl, ta = self._epoch(train_dl, train=True)
            vl, va = self._epoch(val_dl,   train=False)
            self.sched.step()

            self.history["train_loss"].append(tl)
            self.history["train_acc"].append(ta)
            self.history["val_loss"].append(vl)
            self.history["val_acc"].append(va)

            log.info(f"Epoch {ep:03d}/{epochs} | "
                     f"train loss={tl:.4f} acc={ta:.3f} | "
                     f"val loss={vl:.4f} acc={va:.3f}")

            # Early stopping
            if va > best_val:
                best_val, wait = va, 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    log.info(f"Early stopping tại epoch {ep} (best val_acc={best_val:.3f})")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self.history

    def _epoch(self, dl: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for imgs, labels in dl:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if train:
                    self.opt.zero_grad()
                out  = self.model(imgs)
                loss = self.crit(out, labels)
                if train:
                    loss.backward(); self.opt.step()
                total_loss += loss.item() * imgs.size(0)
                correct    += (out.argmax(1) == labels).sum().item()
                total      += imgs.size(0)
        return total_loss/total, correct/total

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        log.info(f"Model đã lưu → {path}")

    def plot_history(self, out_dir: str = "outputs/reports"):
        try:
            import matplotlib.pyplot as plt
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, metric in zip(axes, ["loss", "acc"]):
                ax.plot(self.history[f"train_{metric}"], label="Train")
                ax.plot(self.history[f"val_{metric}"],   label="Val")
                ax.set_title(metric.capitalize()); ax.legend(); ax.grid(True)
            fig.tight_layout()
            path = f"{out_dir}/training_history.png"
            fig.savefig(path, dpi=120)
            log.info(f"Training plot → {path}")
        except Exception as e:
            log.warning(f"Không vẽ được history: {e}")


# ── Runtime Classifier ────────────────────────────────────────────────────────

class BehaviorClassifier:
    """
    Phân loại hành vi tại runtime.
    Tự fallback về rule-based nếu chưa có file weights.

    Sử dụng:
        clf = BehaviorClassifier(cfg.classification)
        label, conf = clf(crop_bgr)
    """

    def __init__(self, cfg):
        self._n       = int(cfg.num_classes           or 5)
        self._thresh  = float(cfg.confidence_threshold or 0.55)
        self._size    = tuple(cfg.input_size)          if cfg.input_size else (224, 224)
        self._model   = None
        self._tf      = get_transforms(self._size, train=False)
        self._load(str(cfg.model or "mobilenet_v3_small"))

    def _load(self, name: str):
        if not _TORCH_OK:
            log.warning("torch không có → dùng rule-based classifier")
            return
        weight_path = Path("models/weights/behavior_clf.pth")
        try:
            m = build_model(name, self._n, pretrained=not weight_path.exists())
            if weight_path.exists():
                m.load_state_dict(torch.load(weight_path, map_location="cpu"))
                log.info(f"Weights loaded ← {weight_path}")
            else:
                log.warning("Chưa có file weights → dùng pretrained backbone (kết quả chưa chính xác)")
            self._model = m.eval()
        except Exception as e:
            log.warning(f"Không load được model ({e}) → rule-based fallback")
            self._model = None

    def __call__(self, crop: np.ndarray) -> Tuple[int, float]:
        if self._model is not None and crop is not None and crop.size > 0:
            return self._nn(crop)
        return self._rule(crop)

    def _nn(self, crop: np.ndarray) -> Tuple[int, float]:
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t   = self._tf(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(t), 1)[0]
        conf, cls = probs.max(0)
        return int(cls), float(conf)

    def _rule(self, crop: Optional[np.ndarray]) -> Tuple[int, float]:
        if crop is None or crop.size == 0:
            return 0, 0.9
        h, w = crop.shape[:2]
        return (2, 0.55) if h / max(w, 1) < 0.7 else (0, 0.88)
