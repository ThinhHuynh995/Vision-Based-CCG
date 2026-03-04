"""
tests/test_all.py — Bộ test toàn diện cho tất cả modules
=========================================================
Chạy: pytest tests/ -v
      pytest tests/ -v -k "Preprocessing"   # chỉ 1 class
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import json
import pytest
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    from src.utils.config import load
    return load("configs/config.yaml")

@pytest.fixture
def frame():
    """480×640 BGR frame ngẫu nhiên."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def small_frame():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

@pytest.fixture
def tmp_video(tmp_path):
    """Video MP4 tổng hợp ngắn (20 frames)."""
    p = str(tmp_path / "test.mp4")
    wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(20):
        wr.write(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
    wr.release()
    return p

@pytest.fixture
def dataset_dir(tmp_path):
    """Dataset thư mục chuẩn train/val/test/<class>/img.jpg."""
    classes = ["Normal", "Fighting", "Falling", "Loitering", "Crowd Panic"]
    for split in ["train", "val", "test"]:
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(10):
                img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"img_{i:03d}.jpg"), img)
    return str(tmp_path)

@pytest.fixture
def labeled_dir(tmp_path):
    """Thư mục ảnh gán nhãn (flat: labeled/<class>/img.jpg)."""
    classes = ["Normal", "Fighting", "Falling", "Loitering", "Crowd Panic"]
    for cls in classes:
        d = tmp_path / cls
        d.mkdir(parents=True)
        for i in range(12):
            img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"img_{i:03d}.jpg"), img)
    return str(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Branch 1 – Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_denoise_gaussian(self, frame):
        from src.preprocessing.processor import denoise
        out = denoise(frame, "gaussian")
        assert out.shape == frame.shape and out.dtype == np.uint8

    def test_denoise_median(self, frame):
        from src.preprocessing.processor import denoise
        assert denoise(frame, "median").shape == frame.shape

    def test_denoise_bilateral(self, frame):
        from src.preprocessing.processor import denoise
        assert denoise(frame, "bilateral").shape == frame.shape

    def test_enhance_clahe(self, frame):
        from src.preprocessing.processor import enhance
        out = enhance(frame, "clahe")
        assert out.shape == frame.shape

    def test_enhance_histogram_eq(self, frame):
        from src.preprocessing.processor import enhance
        assert enhance(frame, "histogram_eq").shape == frame.shape

    def test_enhance_gamma_brightens(self, frame):
        from src.preprocessing.processor import enhance
        # Gamma < 1 làm tối, dùng table với inv_gamma=1/0.7 ≈ 1.43 → làm sáng
        bright = enhance(frame, "gamma")
        assert bright.mean() != frame.mean()   # phải khác nhau

    def test_sharpen_shape(self, frame):
        from src.preprocessing.processor import sharpen
        assert sharpen(frame).shape == frame.shape

    def test_wiener_deblur(self, frame):
        from src.preprocessing.processor import wiener_deblur
        assert wiener_deblur(frame).shape == frame.shape

    def test_pipeline_resize(self, frame, cfg):
        from src.preprocessing.processor import Preprocessor
        pp = Preprocessor(cfg.preprocessing)
        out = pp(frame)
        w, h = cfg.preprocessing.resize
        assert out.shape == (h, w, 3)

    def test_pipeline_none_input(self, cfg):
        from src.preprocessing.processor import Preprocessor
        pp = Preprocessor(cfg.preprocessing)
        assert pp(None) is None

    def test_compare_double_width(self, frame, cfg):
        from src.preprocessing.processor import Preprocessor
        pp  = Preprocessor(cfg.preprocessing)
        cmp = pp.compare(frame)
        w   = cfg.preprocessing.resize[0]
        assert cmp.shape[1] == w * 2


# ─────────────────────────────────────────────────────────────────────────────
# Branch 2 – Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetection:

    def test_det_center(self):
        from src.detection.detector import Det
        d = Det(bbox=(10, 20, 110, 120), conf=0.8)
        assert d.center == (60, 70)
        assert d.w == 100 and d.h == 100

    def test_det_crop(self, frame):
        from src.detection.detector import Det
        d = Det(bbox=(50, 50, 200, 250), conf=0.9)
        crop = d.crop(frame)
        assert crop.shape == (200, 150, 3)

    def test_motion_gate(self, small_frame):
        from src.detection.detector import MotionGate
        mg = MotionGate(min_area=100)
        # Sau vài frame, cần cho bg model khởi động
        for _ in range(5):
            result = mg.has_motion(small_frame)
        assert isinstance(result, bool)

    def test_motion_gate_fg_mask(self, small_frame):
        from src.detection.detector import MotionGate
        mg   = MotionGate()
        mask = mg.fg_mask(small_frame)
        assert mask.shape == small_frame.shape[:2]

    def test_hog_fallback(self, frame):
        from src.detection.detector import PersonDetector
        # Force HOG bằng model name sai
        class FakeCfg:
            model = "not-a-model"
            confidence = 0.3
            iou_threshold = 0.45
            input_size = 640
            motion_filter = False
        dets = PersonDetector(FakeCfg())
        assert isinstance(dets(frame), list)

    def test_density_map_update(self):
        from src.detection.detector import Det, DensityMap
        dm = DensityMap((240, 320))
        dm.update([Det(bbox=(50,50,100,150), conf=0.9)])
        assert dm.get().max() > 0

    def test_density_map_reset(self):
        from src.detection.detector import Det, DensityMap
        dm = DensityMap((240, 320))
        dm.update([Det(bbox=(50,50,100,150), conf=0.9)])
        dm.reset()
        assert dm.get().max() == 0.0

    def test_density_count_zone(self):
        from src.detection.detector import Det, DensityMap
        dm  = DensityMap((240, 320))
        det = Det(bbox=(50,50,100,150), conf=0.9)
        assert dm.count_in_zone((30,30,150,200), [det]) == 1
        assert dm.count_in_zone((200,200,300,240), [det]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Branch 3 – Classification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassification:

    def test_predict_returns_valid(self, cfg):
        from src.classification.classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg.classification)
        lbl, conf = clf(np.random.randint(0, 255, (80, 40, 3), dtype=np.uint8))
        assert 0 <= lbl < 5
        assert 0.0 <= conf <= 1.0

    def test_predict_empty_crop(self, cfg):
        from src.classification.classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg.classification)
        lbl, conf = clf(np.zeros((0, 0, 3), dtype=np.uint8))
        assert isinstance(lbl, int)

    def test_rule_falling_wide_crop(self, cfg):
        """Crop nằm ngang (h/w < 0.7) → rule-based trả về Falling (2)."""
        from src.classification.classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg.classification)
        clf._model = None   # force rule-based
        lbl, _ = clf(np.zeros((30, 120, 3), dtype=np.uint8))
        assert lbl == 2

    def test_build_mobilenet(self):
        import torch
        from src.classification.classifier import build_model
        m   = build_model("mobilenet_v3_small", 5, pretrained=False)
        out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 5)

    def test_build_efficientnet(self):
        import torch
        from src.classification.classifier import build_model
        m   = build_model("efficientnet_b0", 5, pretrained=False)
        out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 5)

    def test_build_resnet(self):
        import torch
        from src.classification.classifier import build_model
        m   = build_model("resnet50", 5, pretrained=False)
        out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 5)

    def test_dataset_loads(self, dataset_dir):
        from src.classification.classifier import BehaviorDataset, get_transforms
        ds = BehaviorDataset(dataset_dir, "train", get_transforms((224,224), True))
        assert len(ds) > 0
        img, lbl = ds[0]
        assert 0 <= lbl < 5

    def test_trainer_one_step(self, dataset_dir):
        import torch
        from torch.utils.data import DataLoader
        from src.classification.classifier import (
            BehaviorDataset, build_model, get_transforms, Trainer
        )
        ds = BehaviorDataset(dataset_dir, "train", get_transforms((64,64), True))
        dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        m  = build_model("mobilenet_v3_small", 5, pretrained=False)
        t  = Trainer(m, device="cpu", lr=1e-3)
        loss, acc = t._epoch(dl, train=True)
        assert loss > 0 and 0.0 <= acc <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Branch 4 – Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestAugmentation:

    def test_augmentor_shape(self, frame):
        from src.augmentation.augmentor import Augmentor
        assert Augmentor()(frame).shape == frame.shape

    def test_mixup_shape_lambda(self, frame):
        from src.augmentation.augmentor import MixUp
        mixed, lam, *_ = MixUp(0.4).mixup(frame, frame, 0, 1)
        assert mixed.shape == frame.shape and 0.0 <= lam <= 1.0

    def test_cutmix_shape_lambda(self, frame):
        from src.augmentation.augmentor import MixUp
        mixed, lam, *_ = MixUp(0.4).cutmix(frame, frame, 0, 2)
        assert mixed.shape == frame.shape and 0.0 <= lam <= 1.0

    def test_grid_dimensions(self, frame):
        from src.augmentation.augmentor import Augmentor
        grid = Augmentor().grid(frame, n=8)
        assert grid.ndim == 3

    def test_augment_dir(self, tmp_path):
        from src.augmentation.augmentor import augment_dir
        src = tmp_path / "Normal"
        src.mkdir()
        for i in range(3):
            cv2.imwrite(str(src / f"img_{i}.jpg"),
                        np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))
        total = augment_dir(str(tmp_path), str(tmp_path / "out"), n=2)
        assert total >= 3   # ít nhất bản gốc được copy


# ─────────────────────────────────────────────────────────────────────────────
# Branch 5 – Tracking & Anomaly
# ─────────────────────────────────────────────────────────────────────────────

class TestTracking:

    def test_iou_identical(self):
        from src.tracking.tracker import _iou
        assert _iou((0,0,10,10), (0,0,10,10)) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from src.tracking.tracker import _iou
        assert _iou((0,0,10,10), (20,20,30,30)) == pytest.approx(0.0)

    def test_iou_partial(self):
        from src.tracking.tracker import _iou
        v = _iou((0,0,10,10), (5,5,15,15))
        assert 0.0 < v < 1.0

    def test_tracker_creates_tracks(self):
        from src.detection.detector import Det
        from src.tracking.tracker import IoUTracker
        class FakeCfg:
            iou_threshold = 0.3
            max_age = 30
            min_hits = 1
        t = IoUTracker(FakeCfg())
        dets = [Det(bbox=(50,50,150,250), conf=0.9)]
        tracks = t.update(dets, fidx=0)
        assert len(tracks) == 1

    def test_tracker_min_hits(self):
        from src.detection.detector import Det
        from src.tracking.tracker import IoUTracker
        class FakeCfg:
            iou_threshold = 0.3
            max_age = 30
            min_hits = 3
        t    = IoUTracker(FakeCfg())
        dets = [Det(bbox=(50,50,150,250), conf=0.9)]
        for fi in range(2):
            t.update(dets, fidx=fi)
        # Chưa đủ min_hits → chưa trả về
        assert len(t.update(dets, fidx=2)) == 1

    def test_tracker_stale_removed(self):
        from src.detection.detector import Det
        from src.tracking.tracker import IoUTracker
        class FakeCfg:
            iou_threshold = 0.3
            max_age = 5
            min_hits = 1
        t = IoUTracker(FakeCfg())
        t.update([Det(bbox=(50,50,150,250), conf=0.9)], fidx=0)
        # Bỏ 10 frame → stale
        assert len(t.update([], fidx=10)) == 0

    def test_track_velocity(self):
        from src.tracking.tracker import Track
        t = Track(tid=1, bbox=(0,0,50,150), center=(25,75))
        t.update((10,10,60,160), (35,85), 0, 0.9, 1)
        t.update((20,20,70,170), (45,95), 0, 0.9, 2)
        assert t.avg_speed > 0

    def test_anomaly_loitering(self):
        from src.tracking.tracker import Track, AnomalyChecker
        class FakeCfg:
            speed_threshold = 80
            loiter_frames = 10
            crowd_threshold = 5
        t = Track(tid=1, bbox=(0,0,50,150), center=(25,75))
        t.stationary = 15
        ac = AnomalyChecker(FakeCfg())
        assert ac.check(t, 0) == "LOITERING"

    def test_anomaly_speeding(self):
        from src.tracking.tracker import Track, AnomalyChecker
        class FakeCfg:
            speed_threshold = 10
            loiter_frames = 60
            crowd_threshold = 5
        t = Track(tid=1, bbox=(0,0,50,150), center=(25,75))
        t.speeds = [200, 200, 200]
        ac = AnomalyChecker(FakeCfg())
        assert ac.check(t, 0) == "RUNNING FAST"

    def test_anomaly_crowd(self):
        from src.tracking.tracker import AnomalyChecker
        class FakeCfg:
            speed_threshold = 80
            loiter_frames = 60
            crowd_threshold = 4
        ac = AnomalyChecker(FakeCfg())
        assert ac.check_crowd(3) is None
        assert ac.check_crowd(5) is not None

    def test_flow_analyzer(self, frame):
        from src.tracking.tracker import FlowAnalyzer
        mag, vis = FlowAnalyzer().analyze(frame, frame)
        assert mag.shape == frame.shape[:2]
        assert vis.shape == frame.shape
        assert mag.mean() < 0.5    # frame giống nhau → không có flow


# ─────────────────────────────────────────────────────────────────────────────
# Dataset – Builder & Validator
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset:

    def test_build_creates_splits(self, labeled_dir, cfg):
        from src.dataset.builder import DatasetBuilder
        out = str(Path(labeled_dir).parent / "out")
        stats = DatasetBuilder(cfg).build(
            labeled_dir, out, val_ratio=0.2, test_ratio=0.1
        )
        assert stats.train > 0
        assert (Path(out) / "train").exists()
        assert (Path(out) / "val").exists()
        assert (Path(out) / "test").exists()

    def test_manifest_written(self, labeled_dir, cfg):
        from src.dataset.builder import DatasetBuilder
        out = str(Path(labeled_dir).parent / "out2")
        DatasetBuilder(cfg).build(labeled_dir, out)
        assert (Path(out) / "manifest.json").exists()
        with open(Path(out) / "manifest.json") as f:
            m = json.load(f)
        assert "train" in m and "val" in m and "test" in m

    def test_dedup_removes_identical(self, tmp_path, cfg):
        from src.dataset.builder import DatasetBuilder
        d = tmp_path / "Normal"
        d.mkdir()
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        # Hai file giống hệt nhau
        cv2.imwrite(str(d / "a.jpg"), img)
        cv2.imwrite(str(d / "b.jpg"), img)
        out   = str(tmp_path / "out")
        stats = DatasetBuilder(cfg).build(str(tmp_path), out,
                                           remove_duplicates=True)
        assert stats.duplicates_rm >= 1

    def test_validate_passes(self, dataset_dir):
        from src.dataset.builder import validate
        report = validate(dataset_dir, min_per_class=1, check_leakage=False)
        assert isinstance(report.passed, bool)

    def test_validate_catches_missing_train(self, tmp_path):
        from src.dataset.builder import validate
        report = validate(str(tmp_path), min_per_class=1)
        assert not report.passed     # train dir không tồn tại → error

    def test_validate_catches_low_count(self, tmp_path):
        from src.dataset.builder import validate
        # Train/Normal có 1 ảnh, min=20 → phải fail
        d = tmp_path / "train" / "Normal"
        d.mkdir(parents=True)
        cv2.imwrite(str(d/"img.jpg"), np.zeros((64,48,3),dtype=np.uint8))
        report = validate(str(tmp_path), min_per_class=20, check_leakage=False)
        assert not report.passed


# ─────────────────────────────────────────────────────────────────────────────
# Draw utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestDraw:

    def test_boxes_shape(self, frame):
        from src.utils.draw import boxes
        out = boxes(frame, [(10,10,100,200)], [1], [0], [0.9])
        assert out.shape == frame.shape

    def test_trajectories_shape(self, frame):
        from src.utils.draw import trajectories
        out = trajectories(frame, {1: [(100,100),(110,110),(120,115)]})
        assert out.shape == frame.shape

    def test_heatmap_nonzero(self, frame):
        from src.utils.draw import heatmap
        acc = np.random.rand(480, 640).astype(np.float32)
        out = heatmap(frame, acc)
        assert out.shape == frame.shape

    def test_heatmap_zero_passthrough(self, frame):
        from src.utils.draw import heatmap
        acc = np.zeros((480, 640), dtype=np.float32)
        out = heatmap(frame, acc)
        assert np.array_equal(out, frame)

    def test_stats_panel_shape(self, frame):
        from src.utils.draw import stats_panel
        out = stats_panel(frame, {"Frame": "5", "Persons": "3"})
        assert out.shape == frame.shape

    def test_alert_banner_shape(self, frame):
        from src.utils.draw import alert_banner
        out = alert_banner(frame, "TEST ALERT")
        assert out.shape == frame.shape


# ─────────────────────────────────────────────────────────────────────────────
# Label Tool (import + basic sanity)
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelTool:

    def test_import(self):
        from tools.label_tool import LabelSession, batch_label, CLASS_NAMES
        assert len(CLASS_NAMES) == 5

    def test_session_creates_dirs(self, tmp_video, tmp_path):
        from tools.label_tool import LabelSession
        out = str(tmp_path / "labeled")
        sess = LabelSession(tmp_video, out, step=2)
        sess.cap.release()
        from src.classification.classifier import CLASS_IDX
        for cls in ["Normal", "Fighting", "Falling", "Loitering", "Crowd Panic"]:
            assert (Path(out) / cls).exists()

    def test_summary_structure(self, tmp_video, tmp_path):
        from tools.label_tool import LabelSession
        sess = LabelSession(str(tmp_video), str(tmp_path / "lb"), step=2)
        sess.labels = {0: 0, 5: 1, 10: 2}
        s = sess.summary()
        assert "video" in s and "labeled_frames" in s and "class_counts" in s
        sess.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (smoke test – không cần torch/yolo)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:

    def test_pipeline_demo_video(self, tmp_video, tmp_path, cfg):
        from src.tracking.tracker import Pipeline
        out     = str(tmp_path / "out.mp4")
        pipe    = Pipeline(cfg)
        summary = pipe.run(tmp_video, out, max_frames=8)
        assert summary["frames_processed"] == 8
        assert "fps_achieved"    in summary
        assert "total_anomalies" in summary
        assert "anomaly_log"     in summary
        assert Path(out).exists()
