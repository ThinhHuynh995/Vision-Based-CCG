"""
src/dataset/builder.py — Dataset Builder & Validator
=====================================================
Chuyển đổi ảnh đã gán nhãn thành cấu trúc train/val/test
sẵn sàng cho PyTorch DataLoader.

Pipeline:
    1. Collect  – Quét toàn bộ ảnh từ data/labeled/
    2. Dedup    – Loại bỏ ảnh trùng (MD5)
    3. Balance  – Oversample class thiếu (tùy chọn)
    4. Split    – Phân chia stratified train/val/test
    5. Write    – Copy ảnh vào data/processed/{split}/{class}/
    6. Validate – Kiểm tra toàn bộ dataset sau khi build

Sử dụng:
    from src.dataset.builder import DatasetBuilder
    builder = DatasetBuilder(cfg)
    stats = builder.build("data/labeled", "data/processed")
    stats.print()

    # Validate riêng:
    from src.dataset.builder import validate
    report = validate("data/processed")
    report.print()
"""
from __future__ import annotations
import cv2
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import src.utils.log as _log

log = _log.get(__name__)

CLASS_NAMES = {0:"Normal", 1:"Fighting", 2:"Falling", 3:"Loitering", 4:"Crowd Panic"}
CLASS_IDX   = {v: k for k, v in CLASS_NAMES.items()}
SPLITS      = ("train", "val", "test")


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Sample:
    path:       Path
    class_id:   int
    class_name: str

    @property
    def md5(self) -> str:
        with open(self.path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


@dataclass
class BuildStats:
    total:          int = 0
    train:          int = 0
    val:            int = 0
    test:           int = 0
    duplicates_rm:  int = 0
    class_dist:     Dict[str, int] = field(default_factory=dict)
    train_dist:     Dict[str, int] = field(default_factory=dict)
    val_dist:       Dict[str, int] = field(default_factory=dict)
    test_dist:      Dict[str, int] = field(default_factory=dict)
    output_dir:     str = ""

    def print(self):
        print("\n" + "═"*55)
        print("  DATASET BUILD SUMMARY")
        print("═"*55)
        print(f"  Tổng mẫu       : {self.total}")
        print(f"  Train          : {self.train}")
        print(f"  Val            : {self.val}")
        print(f"  Test           : {self.test}")
        print(f"  Ảnh trùng xóa : {self.duplicates_rm}")
        print(f"\n  Phân phối Train:")
        for cls, n in sorted(self.train_dist.items()):
            bar = "█" * min(n//2, 30)
            print(f"    {cls:<14} {bar:<30} {n:4d}")
        print(f"\n  Output → {self.output_dir}")
        print("═"*55 + "\n")


@dataclass
class ValidationReport:
    passed:   bool = True
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counts:   Dict[str, Dict[str,int]] = field(default_factory=dict)

    def error(self, msg: str):
        self.errors.append(msg); self.passed = False
        log.error(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)
        log.warning(msg)

    def print(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print("\n" + "═"*55)
        print(f"  VALIDATION  {status}")
        print("═"*55)
        for split, dist in self.counts.items():
            total = sum(dist.values())
            print(f"\n  {split.upper()} ({total} ảnh):")
            for cls, n in sorted(dist.items()):
                bar = "█" * min(n//2, 28)
                print(f"    {cls:<14} {bar:<28} {n:4d}")
        if self.warnings:
            print(f"\n  ⚠  {len(self.warnings)} cảnh báo:")
            for w in self.warnings: print(f"    - {w}")
        if self.errors:
            print(f"\n  ✗  {len(self.errors)} lỗi:")
            for e in self.errors: print(f"    - {e}")
        print("═"*55 + "\n")


# ── Builder ───────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Build dataset từ thư mục ảnh gán nhãn.

    Cấu trúc nguồn chấp nhận:
        data/labeled/
            Normal/       img.jpg ...
            Fighting/     ...
        HOẶC
        data/labeled/
            video1/
                Normal/   ...
            video2/
                Fighting/ ...
    """

    def __init__(self, cfg=None):
        self._seed = int(cfg.seed if cfg and cfg.seed else 42)
        random.seed(self._seed)

    def build(
        self,
        labeled_dir:      str = "data/labeled",
        output_dir:       str = "data/processed",
        val_ratio:        float = 0.15,
        test_ratio:       float = 0.10,
        target_per_class: Optional[int] = None,
        remove_duplicates: bool = True,
        min_size:         Tuple[int,int] = (32, 32),
    ) -> BuildStats:
        """
        Chạy toàn bộ pipeline build.

        Args:
            labeled_dir:       thư mục gốc chứa ảnh gán nhãn
            output_dir:        thư mục output (data/processed)
            val_ratio:         tỉ lệ validation (0–1)
            test_ratio:        tỉ lệ test (0–1)
            target_per_class:  oversample lên N mẫu/class (None = không oversample)
            remove_duplicates: loại bỏ ảnh trùng theo MD5
            min_size:          bỏ ảnh nhỏ hơn (w, h)

        Returns:
            BuildStats
        """
        log.info("=== DatasetBuilder: bắt đầu ===")

        # 1. Thu thập
        samples = self._collect(Path(labeled_dir), min_size)
        log.info(f"Thu thập: {len(samples)} mẫu")

        # 2. Dedup
        n_before = len(samples)
        if remove_duplicates:
            samples = self._dedup(samples)
        dup_rm = n_before - len(samples)
        if dup_rm:
            log.info(f"Đã xóa {dup_rm} ảnh trùng")

        # 3. Balance
        if target_per_class:
            samples = self._balance(samples, target_per_class)
            log.info(f"Sau balance: {len(samples)} mẫu")

        # 4. Split
        splits = self._split(samples, val_ratio, test_ratio)

        # 5. Write
        self._write(splits, Path(output_dir))

        # 6. Stats
        stats = BuildStats(
            total         = len(samples),
            train         = len(splits["train"]),
            val           = len(splits["val"]),
            test          = len(splits["test"]),
            duplicates_rm = dup_rm,
            class_dist    = self._dist(samples),
            train_dist    = self._dist(splits["train"]),
            val_dist      = self._dist(splits["val"]),
            test_dist     = self._dist(splits["test"]),
            output_dir    = output_dir,
        )
        stats.print()

        # Ghi manifest
        self._write_manifest(splits, Path(output_dir))
        return stats

    # ── Private ───────────────────────────────────────────────────────────────

    def _collect(self, root: Path, min_size: Tuple[int,int]) -> List[Sample]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        samples = []
        for cls_name, cls_id in CLASS_IDX.items():
            # Flat: root/ClassName/
            dirs = list(root.glob(cls_name))
            # Nested: root/*/ClassName/
            dirs += list(root.glob(f"*/{cls_name}"))
            for d in dirs:
                if not d.is_dir(): continue
                for p in d.iterdir():
                    if p.suffix.lower() not in exts: continue
                    img = cv2.imread(str(p))
                    if img is None: continue
                    h, w = img.shape[:2]
                    if w < min_size[0] or h < min_size[1]: continue
                    samples.append(Sample(path=p, class_id=cls_id, class_name=cls_name))
        if not samples:
            log.warning(
                f"Không tìm thấy ảnh trong {root}.\n"
                f"  Cấu trúc mong đợi: {root}/{{Normal,Fighting,...}}/img.jpg\n"
                f"  Hoặc: {root}/video1/{{Normal,...}}/img.jpg"
            )
        return samples

    def _dedup(self, samples: List[Sample]) -> List[Sample]:
        seen, out = set(), []
        for s in samples:
            h = s.md5
            if h not in seen:
                seen.add(h); out.append(s)
        return out

    def _balance(self, samples: List[Sample], target: int) -> List[Sample]:
        by_cls: Dict[int, List[Sample]] = defaultdict(list)
        for s in samples:
            by_cls[s.class_id].append(s)
        out = []
        for cls_id in CLASS_IDX.values():
            cls_s = by_cls.get(cls_id, [])
            if not cls_s:
                log.warning(f"Không có mẫu cho class {CLASS_NAMES[cls_id]}")
                continue
            if len(cls_s) >= target:
                out.extend(random.sample(cls_s, target))
            else:
                extra = random.choices(cls_s, k=target - len(cls_s))
                out.extend(cls_s + extra)
        return out

    def _split(
        self, samples: List[Sample], val_r: float, test_r: float
    ) -> Dict[str, List[Sample]]:
        by_cls: Dict[int, List[Sample]] = defaultdict(list)
        for s in samples:
            by_cls[s.class_id].append(s)

        train, val, test = [], [], []
        for cls_s in by_cls.values():
            random.shuffle(cls_s)
            n     = len(cls_s)
            n_tst = max(1, int(n * test_r))
            n_val = max(1, int(n * val_r))
            test.extend(cls_s[:n_tst])
            val.extend( cls_s[n_tst:n_tst+n_val])
            train.extend(cls_s[n_tst+n_val:])

        random.shuffle(train)
        return {"train": train, "val": val, "test": test}

    def _write(self, splits: Dict[str, List[Sample]], out: Path):
        for split, ss in splits.items():
            for s in ss:
                dst_dir = out / split / s.class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / s.path.name
                # Tránh ghi đè
                if dst.exists():
                    i = 1
                    while dst.exists():
                        dst = dst_dir / f"{s.path.stem}_{i}{s.path.suffix}"
                        i += 1
                shutil.copy2(s.path, dst)

    def _dist(self, samples: List[Sample]) -> Dict[str, int]:
        return dict(Counter(s.class_name for s in samples))

    def _write_manifest(self, splits: Dict[str, List[Sample]], out: Path):
        manifest = {
            sp: [{"file": s.path.name, "class": s.class_name, "id": s.class_id}
                 for s in ss]
            for sp, ss in splits.items()
        }
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        log.info(f"Manifest → {out/'manifest.json'}")


# ── Validate (standalone) ─────────────────────────────────────────────────────

def validate(
    dataset_dir:     str,
    min_per_class:   int   = 20,
    blur_thr:        float = 60.0,
    dark_thr:        float = 25.0,
    check_leakage:   bool  = True,
) -> ValidationReport:
    """
    Kiểm tra toàn bộ dataset sau khi build.

    Các kiểm tra:
        - Thư mục train/val/test tồn tại
        - Đủ ảnh tối thiểu mỗi class
        - Phát hiện class mất cân bằng nghiêm trọng
        - Ảnh bị hỏng / không đọc được
        - Ảnh quá mờ (Laplacian variance < blur_thr)
        - Ảnh quá tối (mean < dark_thr)
        - Data leakage: cùng ảnh xuất hiện ở nhiều split

    Args:
        dataset_dir:   đường dẫn đến data/processed/
        min_per_class: số ảnh tối thiểu mỗi class trong train
        blur_thr:      ngưỡng Laplacian variance (ảnh dưới = mờ)
        dark_thr:      ngưỡng độ sáng trung bình (ảnh dưới = tối)
        check_leakage: bật kiểm tra MD5 cross-split

    Returns:
        ValidationReport
    """
    root   = Path(dataset_dir)
    report = ValidationReport()
    exts   = {".jpg", ".jpeg", ".png", ".bmp"}

    split_hashes: Dict[str, Dict[str, str]] = {}   # split → {path_str: md5}
    total_img = corrupt = blurry = dark = 0

    for split in SPLITS:
        sd = root / split
        if not sd.exists():
            (report.error if split == "train" else report.warn)(
                f"Thiếu thư mục: {sd}"
            )
            continue

        hashes: Dict[str, str] = {}
        split_counts: Dict[str, int] = {}

        for cls_name in CLASS_NAMES.values():
            cd = sd / cls_name
            if not cd.exists():
                report.warn(f"Thiếu class dir: {split}/{cls_name}")
                split_counts[cls_name] = 0
                continue

            imgs = [p for p in cd.iterdir() if p.suffix.lower() in exts]
            split_counts[cls_name] = len(imgs)
            total_img += len(imgs)

            for p in imgs:
                # Corrupt check
                img = cv2.imread(str(p))
                if img is None:
                    report.warn(f"Ảnh hỏng: {p.relative_to(root)}")
                    corrupt += 1
                    continue

                # Leakage hash
                with open(p, "rb") as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
                hashes[str(p)] = md5

                # Blur
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thr:
                    blurry += 1

                # Dark
                if gray.mean() < dark_thr:
                    dark += 1

        report.counts[split] = split_counts
        split_hashes[split]  = hashes

        # Class balance check (train only)
        if split == "train" and split_counts:
            max_n = max(split_counts.values()) if split_counts else 1
            for cls_name, n in split_counts.items():
                if n < min_per_class:
                    report.error(
                        f"Quá ít mẫu train — {cls_name}: {n} "
                        f"(cần ≥ {min_per_class})"
                    )
                elif max_n > 0 and n / max_n < 0.15:
                    report.warn(
                        f"Mất cân bằng nghiêm trọng — {cls_name}: "
                        f"{n} vs max {max_n} ({n/max_n*100:.0f}%)"
                    )

    # Cross-split leakage
    if check_leakage:
        sp_list = list(split_hashes.items())
        for i in range(len(sp_list)):
            for j in range(i+1, len(sp_list)):
                s1, h1 = sp_list[i]
                s2, h2 = sp_list[j]
                leak   = set(h1.values()) & set(h2.values())
                if leak:
                    report.warn(
                        f"Data leakage: {len(leak)} ảnh trùng giữa "
                        f"{s1} và {s2}"
                    )

    # Quality summary warnings
    if total_img > 0:
        if blurry / total_img > 0.20:
            report.warn(
                f"{blurry/total_img*100:.1f}% ảnh có thể mờ "
                f"(Laplacian < {blur_thr})"
            )
        if dark / total_img > 0.15:
            report.warn(
                f"{dark/total_img*100:.1f}% ảnh có thể quá tối "
                f"(mean < {dark_thr})"
            )
        if corrupt:
            report.error(f"{corrupt} ảnh bị hỏng không đọc được")

    report.print()
    return report
