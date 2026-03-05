"""
src/autolabel/autolabeler.py — Auto-Labeling Engine
=====================================================

Hai chế độ hoạt động:

  MODE 1: ZERO-SHOT  (chưa có data gì)
  ─────────────────────────────────────
  Dùng CLIP (openai/clip) so sánh cosine similarity giữa
  frame/crop với text prompt cho từng class.
  Không cần train, không cần label thủ công.

  MODE 2: SEMI-SUPERVISED  (đã có ~50+ ảnh/class)
  ─────────────────────────────────────────────────
  Dùng model đã train một phần (behavior_clf.pth) để
  tự gán nhãn, chỉ giữ lại những frame confidence cao.
  Frame confidence thấp → queue cho người review.

Sử dụng:
    from src.autolabel.autolabeler import AutoLabeler
    al = AutoLabeler.from_config(cfg)
    results = al.label_video("data/raw/cctv.mp4", out_dir="data/labeled")
    print(f"Auto: {results.auto_count}  Cần review: {results.review_count}")
"""
from __future__ import annotations
import cv2
import numpy as np
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import src.utils.log as _log

log = _log.get(__name__)

CLASS_NAMES = {0:"Normal", 1:"Fighting", 2:"Falling", 3:"Loitering", 4:"Crowd Panic"}

# Text prompts cho từng class — CLIP sẽ match với những mô tả này
CLIP_PROMPTS: Dict[int, List[str]] = {
    0: [
        "a person walking normally in a surveillance camera",
        "normal human activity in CCTV footage",
        "people standing or walking calmly",
    ],
    1: [
        "two people fighting or punching each other",
        "violent altercation between people in security footage",
        "people involved in a physical fight",
    ],
    2: [
        "a person falling down on the ground",
        "someone collapsed or fallen on floor",
        "a person who has fallen and is lying down",
    ],
    3: [
        "a person standing still for a long time loitering",
        "someone lingering suspiciously in one place",
        "a person loitering not moving in surveillance footage",
    ],
    4: [
        "a crowd of people panicking and running",
        "mass crowd panic and stampede",
        "many people running in fear in a crowd",
    ],
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    frame_idx: int
    label_id:  int
    label_name: str
    confidence: float
    auto_accepted: bool    # True = tự động lưu; False = cần review


@dataclass
class LabelingResult:
    video_path:    str
    out_dir:       str
    total_frames:  int
    sampled_frames: int
    auto_count:    int
    review_count:  int
    skipped_count: int       # frame bị bỏ qua (không phát hiện người)
    results:       List[FrameResult] = field(default_factory=list)
    mode:          str = "unknown"

    def print(self):
        print("\n" + "═"*55)
        print(f"  AUTO-LABEL KẾT QUẢ  [{self.mode.upper()}]")
        print("═"*55)
        print(f"  Video            : {Path(self.video_path).name}")
        print(f"  Frames lấy mẫu  : {self.sampled_frames}/{self.total_frames}")
        print(f"  ✅ Tự động lưu   : {self.auto_count}")
        print(f"  👁  Cần review    : {self.review_count}")
        print(f"  ⏭  Bỏ qua        : {self.skipped_count}")

        from collections import Counter
        auto = [r for r in self.results if r.auto_accepted]
        cnt  = Counter(r.label_name for r in auto)
        if cnt:
            print(f"\n  Phân phối tự động:")
            for cls, n in sorted(cnt.items()):
                bar = "█" * min(n, 30)
                print(f"    {cls:<14} {bar:<30} {n:4d}")

        print(f"\n  Output → {self.out_dir}")
        print("═"*55 + "\n")

    def save_report(self, path: str):
        data = {
            "video":          self.video_path,
            "mode":           self.mode,
            "total_frames":   self.total_frames,
            "sampled_frames": self.sampled_frames,
            "auto_count":     self.auto_count,
            "review_count":   self.review_count,
            "skipped_count":  self.skipped_count,
            "frames": [
                {"idx": r.frame_idx, "label": r.label_name,
                 "conf": round(r.confidence, 4), "auto": r.auto_accepted}
                for r in self.results
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"Report → {path}")


# ── CLIP Backend ──────────────────────────────────────────────────────────────

class CLIPBackend:
    """
    Zero-shot classification với CLIP.
    Thử theo thứ tự: open_clip → transformers CLIP → rule-based fallback.
    """

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._encoder = None
        self._kind    = "rule"
        self._text_features: Optional[np.ndarray] = None
        self._init()

    def _init(self):
        # 1. Thử open_clip (nhẹ hơn, không cần torchvision)
        try:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model = model.eval().to(self.device)
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            # Pre-encode text prompts
            with torch.no_grad():
                all_feats = []
                for cls_id in sorted(CLASS_NAMES):
                    prompts = CLIP_PROMPTS[cls_id]
                    tokens  = tokenizer(prompts).to(self.device)
                    feats   = model.encode_text(tokens)
                    feats  /= feats.norm(dim=-1, keepdim=True)
                    all_feats.append(feats.mean(0))   # average prompts
                self._text_feats = torch.stack(all_feats)  # (5, D)

            self._model      = model
            self._preprocess = preprocess
            self._tokenizer  = tokenizer
            self._kind       = "open_clip"
            log.info("CLIP backend: open_clip ViT-B-32")
            return
        except ImportError:
            pass

        # 2. Thử transformers CLIP
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).eval().to(self.device)
            clip_proc  = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            all_text = [p for cid in sorted(CLASS_NAMES) for p in CLIP_PROMPTS[cid]]
            inputs   = clip_proc(text=all_text, return_tensors="pt", padding=True)
            inputs   = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            # Average per class
            n = len(CLIP_PROMPTS[0])
            self._text_feats = torch.stack([
                feats[i*n:(i+1)*n].mean(0)
                for i in range(len(CLASS_NAMES))
            ])
            self._model      = clip_model
            self._processor  = clip_proc
            self._kind       = "transformers_clip"
            log.info("CLIP backend: transformers CLIP ViT-B-32")
            return
        except ImportError:
            pass

        # 3. Fallback: rule-based (không cần gì)
        log.warning(
            "CLIP không khả dụng (pip install open-clip-torch hoặc transformers)\n"
            "  → Dùng rule-based fallback (ít chính xác hơn)"
        )
        self._kind = "rule"

    def encode_image(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Trả về image embedding hoặc None nếu không có CLIP."""
        if self._kind == "rule":
            return None

        import torch
        from PIL import Image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            if self._kind == "open_clip":
                t = self._preprocess(pil).unsqueeze(0).to(self.device)
                f = self._model.encode_image(t)
            else:
                inp = self._processor(images=pil, return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                f   = self._model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        return f

    def predict(self, img_bgr: np.ndarray) -> Tuple[int, float]:
        """Trả về (class_id, confidence)."""
        if self._kind == "rule":
            return self._rule_predict(img_bgr)

        import torch
        f = self.encode_image(img_bgr)
        if f is None:
            return self._rule_predict(img_bgr)

        sims = (f @ self._text_feats.T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=0)
        cls  = int(probs.argmax())
        conf = float(probs[cls])
        return cls, conf

    def _rule_predict(self, img: np.ndarray) -> Tuple[int, float]:
        """Heuristic đơn giản không cần model."""
        if img is None or img.size == 0:
            return 0, 0.6
        h, w = img.shape[:2]
        ratio = h / max(w, 1)
        # Nằm ngang → Falling
        if ratio < 0.7:
            return 2, 0.62
        # Quá nhiều người (frame rộng) → Crowd Panic heuristic
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if edges.sum() / edges.size > 0.08:
            return 1, 0.58   # nhiều edge → có thể Fighting/Panic
        return 0, 0.70       # mặc định Normal

    @property
    def mode(self) -> str:
        return self._kind


# ── Auto-Labeler ──────────────────────────────────────────────────────────────

class AutoLabeler:
    """
    Tự động gán nhãn video CCTV.

    Sử dụng:
        al = AutoLabeler(cfg, mode="zero_shot")
        result = al.label_video("data/raw/cctv.mp4", "data/labeled")
        result.print()

        # Hoặc semi-supervised:
        al = AutoLabeler(cfg, mode="semi")
        result = al.label_video(...)
    """

    def __init__(self, cfg, mode: str = "auto"):
        """
        Args:
            cfg:  project config
            mode: "auto"        – chọn tự động (semi nếu có weights, zero_shot nếu không)
                  "zero_shot"   – luôn dùng CLIP
                  "semi"        – luôn dùng model đã train
        """
        self._cfg        = cfg
        self._auto_thr   = 0.75    # confidence ≥ này → tự động lưu
        self._review_thr = 0.50    # confidence < này → bỏ qua
        self._device     = str(cfg.device or "cpu")
        self._mode       = self._resolve_mode(mode)
        self._clip       = None
        self._clf_model  = None
        self._clf_tf     = None
        self._setup()

    def _resolve_mode(self, mode: str) -> str:
        if mode != "auto":
            return mode
        weight_path = Path(self._cfg.paths.model_weights) / "behavior_clf.pth"
        return "semi" if weight_path.exists() else "zero_shot"

    def _setup(self):
        if self._mode == "zero_shot":
            log.info("AutoLabeler: ZERO-SHOT mode (CLIP)")
            self._clip = CLIPBackend(self._device)
        else:
            log.info("AutoLabeler: SEMI-SUPERVISED mode (fine-tuned model)")
            self._load_clf()

    def _load_clf(self):
        try:
            import torch
            from src.classification.classifier import build_model, get_transforms
            weight_path = Path(self._cfg.paths.model_weights) / "behavior_clf.pth"
            clf_cfg     = self._cfg.classification
            model = build_model(
                str(clf_cfg.model or "mobilenet_v3_small"),
                int(clf_cfg.num_classes or 5),
                pretrained=False,
            )
            model.load_state_dict(
                torch.load(str(weight_path), map_location="cpu")
            )
            model.eval()
            self._clf_model = model
            self._clf_tf    = get_transforms(
                tuple(clf_cfg.input_size or [224,224]), train=False
            )
            log.info(f"Semi model loaded ← {weight_path}")
        except Exception as e:
            log.warning(f"Semi-supervised model failed ({e}) → fallback to CLIP")
            self._mode = "zero_shot"
            self._clip = CLIPBackend(self._device)

    # ── Main entry point ──────────────────────────────────────────────────────

    def label_video(
        self,
        video_path: str,
        out_dir:    str,
        frame_step: int = 10,
        max_frames: Optional[int] = None,
    ) -> LabelingResult:
        """
        Xử lý toàn bộ video, tự gán nhãn từng frame.

        Args:
            video_path:  đường dẫn video CCTV
            out_dir:     thư mục output (cùng cấu trúc như label_tool)
            frame_step:  lấy mẫu 1 frame mỗi N frame
            max_frames:  giới hạn số frame lấy mẫu (None = hết video)

        Returns:
            LabelingResult với thống kê chi tiết
        """
        from src.detection.detector import PersonDetector, MotionGate

        video_path = str(video_path)
        out_dir    = Path(out_dir)

        # Tạo thư mục class
        for cls_name in CLASS_NAMES.values():
            (out_dir / cls_name).mkdir(parents=True, exist_ok=True)
        review_dir = out_dir / "_review"
        review_dir.mkdir(exist_ok=True)

        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")

        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps           = cap.get(cv2.CAP_PROP_FPS) or 25
        det           = PersonDetector(self._cfg.detection)
        motion_gate   = MotionGate(min_area=600)
        labels_dict   = {}
        results       = []
        auto_count    = review_count = skipped = 0
        sampled       = 0
        t0            = time.time()

        log.info(f"Auto-labeling: {video_path}  "
                 f"({total_frames} frames, step={frame_step}, mode={self._mode})")

        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fidx += 1
            if fidx % frame_step != 0:
                continue
            if max_frames and sampled >= max_frames:
                break

            sampled += 1

            # Bỏ qua frame tĩnh
            if not motion_gate.has_motion(frame):
                skipped += 1
                continue

            # Phát hiện người
            dets = det(frame)
            if not dets:
                skipped += 1
                continue

            # Lấy crop lớn nhất (person nổi bật nhất)
            best = max(dets, key=lambda d: d.w * d.h)
            crop = best.crop(frame)
            if crop.size == 0:
                skipped += 1
                continue

            # Phân loại
            cls_id, conf = self._predict(crop)

            # Quyết định
            if conf >= self._auto_thr:
                # Tự động lưu
                cls_name = CLASS_NAMES[cls_id]
                fname    = f"{Path(video_path).stem}_f{fidx:06d}.jpg"
                cv2.imwrite(str(out_dir / cls_name / fname), crop)
                labels_dict[fidx] = cls_id
                auto_count += 1
                accepted = True
            elif conf >= self._review_thr:
                # Lưu vào _review/ với tên chứa nhãn đề xuất
                cls_name = CLASS_NAMES[cls_id]
                fname = f"{Path(video_path).stem}_f{fidx:06d}_{cls_name}_{conf:.2f}.jpg"
                cv2.imwrite(str(review_dir / fname), crop)
                review_count += 1
                accepted = False
            else:
                skipped += 1
                accepted = False
                cls_id, conf = 0, 0.0   # không đủ tin cậy

            results.append(FrameResult(
                frame_idx     = fidx,
                label_id      = cls_id,
                label_name    = CLASS_NAMES[cls_id],
                confidence    = conf,
                auto_accepted = accepted,
            ))

            # Progress log mỗi 100 frame
            if sampled % 100 == 0:
                elapsed = time.time() - t0
                fps_proc = sampled / max(elapsed, 0.001)
                log.info(f"  {sampled} frames | auto={auto_count} "
                         f"review={review_count} | {fps_proc:.1f}fps")

        cap.release()

        # Lưu labels.json
        labels_file = out_dir / "labels.json"
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(labels_dict, f, indent=2, ensure_ascii=False)

        result = LabelingResult(
            video_path    = video_path,
            out_dir       = str(out_dir),
            total_frames  = total_frames,
            sampled_frames = sampled,
            auto_count    = auto_count,
            review_count  = review_count,
            skipped_count = skipped,
            results       = results,
            mode          = self._mode,
        )
        result.print()

        # Lưu report
        report_path = str(
            Path(self._cfg.paths.output_reports) /
            f"autolabel_{Path(video_path).stem}.json"
        )
        result.save_report(report_path)
        return result

    def label_folder(
        self,
        video_folder: str,
        out_dir:      str,
        frame_step:   int = 10,
    ) -> List[LabelingResult]:
        """Auto-label tất cả video trong folder."""
        folder  = Path(video_folder)
        videos  = (sorted(folder.glob("*.mp4")) +
                   sorted(folder.glob("*.avi")) +
                   sorted(folder.glob("*.MP4")) +
                   sorted(folder.glob("*.AVI")))
        if not videos:
            log.error(f"Không tìm thấy video trong {folder}")
            return []

        log.info(f"Auto-labeling {len(videos)} videos...")
        results = []
        for i, v in enumerate(videos, 1):
            log.info(f"[{i}/{len(videos)}] {v.name}")
            r = self.label_video(
                str(v),
                str(Path(out_dir) / v.stem),
                frame_step=frame_step,
            )
            results.append(r)

        # Tổng kết
        total_auto   = sum(r.auto_count for r in results)
        total_review = sum(r.review_count for r in results)
        print(f"\n{'═'*50}")
        print(f"  TỔNG KẾT {len(videos)} VIDEOS")
        print(f"  Auto: {total_auto}   Cần review: {total_review}")
        print(f"{'═'*50}\n")
        return results

    # ── Predict backend ───────────────────────────────────────────────────────

    def _predict(self, crop: np.ndarray) -> Tuple[int, float]:
        if self._mode == "zero_shot" and self._clip:
            return self._clip.predict(crop)
        elif self._clf_model is not None:
            return self._semi_predict(crop)
        elif self._clip:
            return self._clip.predict(crop)
        return 0, 0.5

    def _semi_predict(self, crop: np.ndarray) -> Tuple[int, float]:
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t   = self._clf_tf(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._clf_model(t), 1)[0]
        conf, cls = probs.max(0)
        return int(cls), float(conf)

    # ── Config factory ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg, mode: str = "auto") -> "AutoLabeler":
        return cls(cfg, mode=mode)
