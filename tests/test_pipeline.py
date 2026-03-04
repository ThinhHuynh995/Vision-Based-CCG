"""tests/test_pipeline.py — Full unit tests. Run: pytest tests/ -v"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2, json, pytest
from pathlib import Path

@pytest.fixture
def frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def cfg():
    from src.utils.config_loader import load_config
    return load_config("configs/config.yaml")

@pytest.fixture
def tmp_video(tmp_path):
    p = str(tmp_path / "test.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320,240))
    for _ in range(20):
        w.write(np.random.randint(0,255,(240,320,3),dtype=np.uint8))
    w.release()
    return p

@pytest.fixture
def dataset_dir(tmp_path):
    class_names = ["Normal","Fighting","Falling","Loitering","Crowd Panic"]
    for split in ["train","val","test"]:
        for cls in class_names:
            d = tmp_path / split / cls; d.mkdir(parents=True)
            for i in range(8):
                cv2.imwrite(str(d/f"img_{i:03d}.jpg"),
                            np.random.randint(0,255,(64,48,3),dtype=np.uint8))
    return str(tmp_path)

# Branch 1
class TestPreprocessing:
    def test_gaussian(self, frame):
        from src.preprocessing.image_processor import denoise_gaussian
        assert denoise_gaussian(frame).shape == frame.shape
    def test_clahe(self, frame):
        from src.preprocessing.image_processor import enhance_clahe
        assert enhance_clahe(frame).shape == frame.shape
    def test_gamma_bright(self, frame):
        from src.preprocessing.image_processor import enhance_gamma
        assert enhance_gamma(frame, 2.0).mean() > frame.mean()
    def test_wiener(self, frame):
        from src.preprocessing.image_processor import wiener_deblur
        assert wiener_deblur(frame).shape == frame.shape
    def test_pipeline(self, frame, cfg):
        from src.preprocessing.image_processor import ImagePreprocessor
        pp = ImagePreprocessor(cfg["preprocessing"])
        out = pp.process(frame)
        w,h = cfg["preprocessing"]["resize"]
        assert out.shape == (h,w,3)
    def test_compare(self, frame, cfg):
        from src.preprocessing.image_processor import ImagePreprocessor
        pp = ImagePreprocessor(cfg["preprocessing"])
        comp = pp.compare(frame)
        assert comp.shape[1] == pp.resize_wh[0] * 2

# Branch 2
class TestDetection:
    def test_detection_attrs(self):
        from src.detection.detector import Detection
        d = Detection(bbox=(10,20,110,120), confidence=0.8)
        assert d.center == (60,70) and d.area == 10000
    def test_hog(self, frame):
        from src.detection.detector import PersonDetector
        assert isinstance(PersonDetector({"model":"none","confidence":0.3,"iou_threshold":0.45}).detect(frame), list)
    def test_mog2(self):
        from src.detection.detector import MotionDetector
        md = MotionDetector("mog2")
        f = np.zeros((240,320,3),dtype=np.uint8)
        for _ in range(3): mask,boxes = md.detect(f)
        assert mask.shape == (240,320)
    def test_density(self):
        from src.detection.detector import Detection, DensityEstimator
        de = DensityEstimator((240,320)); de.update([Detection(bbox=(50,50,100,150),confidence=0.9)])
        assert de.get_heatmap().max() > 0
    def test_density_reset(self):
        from src.detection.detector import Detection, DensityEstimator
        de = DensityEstimator((240,320)); de.update([Detection(bbox=(50,50,100,150),confidence=0.9)])
        de.reset(); assert de.get_heatmap().max() == 0.0
    def test_count_zone(self):
        from src.detection.detector import Detection, DensityEstimator
        de = DensityEstimator((240,320))
        d = Detection(bbox=(50,50,100,150),confidence=0.9)
        assert de.count_in_zone((30,30,150,200),[d]) == 1
        assert de.count_in_zone((200,200,300,240),[d]) == 0

# Branch 3
class TestClassification:
    def test_predict(self, cfg):
        from src.classification.behavior_classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg["classification"])
        lbl,conf = clf.predict(np.random.randint(0,255,(80,40,3),dtype=np.uint8))
        assert 0<=lbl<5 and 0.0<=conf<=1.0
    def test_predict_empty(self, cfg):
        from src.classification.behavior_classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg["classification"])
        lbl,_ = clf.predict(np.zeros((0,0,3),dtype=np.uint8))
        assert isinstance(lbl,int)
    def test_rule_falling(self, cfg):
        from src.classification.behavior_classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg["classification"]); clf.model = None
        lbl,_ = clf.predict(np.zeros((30,120,3),dtype=np.uint8))
        assert lbl == 2
    def test_batch(self, cfg):
        from src.classification.behavior_classifier import BehaviorClassifier
        clf = BehaviorClassifier(cfg["classification"])
        res = clf.predict_batch([np.random.randint(0,255,(64,40,3),dtype=np.uint8) for _ in range(3)])
        assert len(res) == 3

# Branch 4
class TestAugmentation:
    def test_shape(self, frame):
        from src.augmentation.augmentor import ImageAugmentor
        assert ImageAugmentor({}).augment(frame).shape == frame.shape
    def test_batch(self, frame):
        from src.augmentation.augmentor import ImageAugmentor
        assert len(ImageAugmentor({}).augment_batch([frame,frame],n_per_image=3)) == 6
    def test_mixup(self, frame):
        from src.augmentation.augmentor import MixUpAugmentor
        mixed,lam,_,_ = MixUpAugmentor(0.4).mixup(frame,frame,0,1)
        assert 0<=lam<=1 and mixed.shape==frame.shape
    def test_cutmix(self, frame):
        from src.augmentation.augmentor import MixUpAugmentor
        mixed,lam,_,_ = MixUpAugmentor(0.4).cutmix(frame,frame,0,2)
        assert 0<=lam<=1 and mixed.shape==frame.shape
    def test_generate_dataset(self, tmp_path):
        from src.augmentation.augmentor import generate_augmented_dataset
        src = tmp_path/"src"/"Normal"; src.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(src/f"img_{i}.jpg"),np.random.randint(0,255,(64,48,3),dtype=np.uint8))
        generate_augmented_dataset(str(tmp_path/"src"), str(tmp_path/"dst"), n_per_image=2)
        assert len(list((tmp_path/"dst").rglob("*.jpg"))) >= 3

# Branch 5
class TestTracking:
    def test_iou(self):
        from src.tracking.tracker import _iou
        assert _iou((0,0,10,10),(0,0,10,10)) == pytest.approx(1.0)
        assert _iou((0,0,10,10),(20,20,30,30)) == pytest.approx(0.0)
        assert 0 < _iou((0,0,10,10),(5,5,15,15)) < 1
    def test_tracker(self):
        from src.detection.detector import Detection
        from src.tracking.tracker import SimpleTracker
        t = SimpleTracker({"iou_threshold":0.3,"max_age":30,"min_hits":1})
        dets = [Detection(bbox=(50,50,150,250),confidence=0.9)]
        assert len(t.update(dets,frame_idx=0)) == 1
    def test_min_hits(self):
        from src.detection.detector import Detection
        from src.tracking.tracker import SimpleTracker
        t = SimpleTracker({"iou_threshold":0.3,"max_age":30,"min_hits":3})
        d = [Detection(bbox=(50,50,150,250),confidence=0.9)]
        for fi in range(2): t.update(d,frame_idx=fi)
        assert len(t.update(d,frame_idx=2)) == 1
    def test_stale(self):
        from src.detection.detector import Detection
        from src.tracking.tracker import SimpleTracker
        t = SimpleTracker({"iou_threshold":0.3,"max_age":5,"min_hits":1})
        t.update([Detection(bbox=(50,50,150,250),confidence=0.9)],frame_idx=0)
        assert len(t.update([],frame_idx=10)) == 0
    def test_loitering(self):
        from src.tracking.tracker import AnomalyDetector, TrackState
        ad = AnomalyDetector({"speed_threshold":80,"loiter_frames":10,"crowd_density_threshold":5})
        ts = TrackState(track_id=1,bbox=(0,0,50,150),center=(25,75))
        ts.frames_stationary = 15
        assert ad.check_track(ts,100) == "LOITERING"
    def test_speeding(self):
        from src.tracking.tracker import AnomalyDetector, TrackState
        ad = AnomalyDetector({"speed_threshold":10,"loiter_frames":60,"crowd_density_threshold":5})
        ts = TrackState(track_id=1,bbox=(0,0,50,150),center=(25,75))
        ts.velocities = [200,200,200]
        assert ad.check_track(ts,100) == "RUNNING FAST"
    def test_crowd(self):
        from src.tracking.tracker import AnomalyDetector
        ad = AnomalyDetector({"speed_threshold":80,"loiter_frames":60,"crowd_density_threshold":3})
        assert ad.check_crowd(2) is None and ad.check_crowd(5) is not None
    def test_optical_flow(self, frame):
        from src.tracking.tracker import OpticalFlowAnalyzer
        mag,vis = OpticalFlowAnalyzer().analyze(frame,frame)
        assert mag.shape == frame.shape[:2] and vis.shape == frame.shape

# IO
class TestVideoSource:
    def test_open(self, tmp_video):
        from src.io.video_source import VideoSource
        with VideoSource(tmp_video) as src:
            assert src.is_file and src.fps > 0 and src.total_frames > 0
    def test_read_frames(self, tmp_video):
        from src.io.video_source import VideoSource
        with VideoSource(tmp_video) as src:
            frames = list(src.read_frames(max_frames=5))
        assert len(frames) == 5

# Dataset
class TestDataset:
    def test_build(self, dataset_dir, cfg):
        from src.dataset.builder import DatasetBuilder
        out = str(Path(dataset_dir).parent/"out")
        stats = DatasetBuilder(cfg).build(dataset_dir, out, val_ratio=0.2, test_ratio=0.1)
        assert stats.train_count > 0 and (Path(out)/"train").exists()
    def test_validate_passes(self, dataset_dir):
        from src.dataset.validator import DatasetValidator
        r = DatasetValidator.run(dataset_dir, min_samples=1)
        assert isinstance(r.passed, bool)
    def test_validate_missing_train(self, tmp_path):
        from src.dataset.validator import DatasetValidator
        r = DatasetValidator.run(str(tmp_path), min_samples=1)
        assert not r.passed

# Reporting
class TestReporting:
    def _summary(self):
        from src.reporting.reporter import RunSummary
        return RunSummary(
            video_path="test.mp4", frames_processed=500,
            fps_achieved=12.3, elapsed_sec=40.6,
            anomaly_log=[
                {"time":"10:00:01","frame":50, "track_id":1,"type":"FIGHTING","center":(100,200)},
                {"time":"10:00:05","frame":120,"track_id":2,"type":"LOITERING","center":(300,100)},
            ],
        )
    def test_csv(self, tmp_path):
        from src.reporting.reporter import Reporter
        p = Reporter(self._summary(), str(tmp_path)).export_csv("log.csv")
        assert p.exists() and len(p.read_text().splitlines()) == 3
    def test_json(self, tmp_path):
        from src.reporting.reporter import Reporter
        p = Reporter(self._summary(), str(tmp_path)).export_json()
        data = json.loads(p.read_text())
        assert data["frames_processed"] == 500
    def test_txt_fallback(self, tmp_path):
        from src.reporting.reporter import Reporter
        r = Reporter(self._summary(), str(tmp_path))
        txt = tmp_path/"report.txt"; r._export_txt(txt)
        assert "FIGHTING" in txt.read_text()

# Visualizer
class TestVisualizer:
    def test_draw_detections(self, frame):
        from src.utils.visualizer import draw_detections
        out = draw_detections(frame,[(10,10,100,200)],[1],[0],[0.9])
        assert out.shape == frame.shape
    def test_trajectories(self, frame):
        from src.utils.visualizer import draw_trajectories
        out = draw_trajectories(frame,{1:[(100,100),(110,110)]})
        assert out.shape == frame.shape
    def test_heatmap(self, frame):
        from src.utils.visualizer import build_heatmap
        hm = np.random.rand(480,640).astype(np.float32)
        assert build_heatmap(frame,hm).shape == frame.shape
    def test_heatmap_empty(self, frame):
        from src.utils.visualizer import build_heatmap
        hm = np.zeros((480,640),dtype=np.float32)
        assert np.array_equal(build_heatmap(frame,hm), frame)
    def test_put_stats(self, frame):
        from src.utils.visualizer import put_stats
        out = put_stats(frame,{"Frame":"5","Persons":"3"})
        assert out.shape == frame.shape
