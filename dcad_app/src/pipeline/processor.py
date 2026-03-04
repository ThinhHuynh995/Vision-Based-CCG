import cv2
import numpy as np
from pathlib import Path
import json
import time
import random
from datetime import datetime
from PIL import Image, ImageEnhance
import math

from src.training.autoencoder import PCAAutoencoder
from src.models.ccg_gate import CCGGate
from src.training.feature_extractor import FeatureExtractor


class MediaProcessor:
    ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"}

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.feature_extractor = FeatureExtractor()
        self.ae = None
        self.ccg = None
        self.model_name = "simulation"
        self._load_active_model()

    def _load_active_model(self):
        registry_path = self.data_dir / "models" / "registry.json"
        if not registry_path.exists():
            self.ae = None; self.ccg = None; self.model_name = "simulation"; return
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        active_id = registry.get("active_session")
        if not active_id:
            self.ae = None; self.ccg = None; self.model_name = "simulation"; return
        session = next((s for s in registry.get("sessions", []) if s.get("session_id") == active_id), None)
        if not session:
            self.ae = None; self.ccg = None; self.model_name = "simulation"; return
        self.ae = PCAAutoencoder(); self.ae.load(session["files"]["autoencoder"])
        self.ccg = CCGGate(); self.ccg.load(session["files"]["ccg"])
        self.model_name = f"pca_{active_id[:6]}"

    def _extract_media_frame(self, file_path: Path):
        ext = file_path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(file_path))
            return img, "image"
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 3))
        ok, frame = cap.read(); cap.release()
        if not ok:
            raise ValueError("Không đọc được frame video")
        return frame, "video"

    def _save_img(self, path: Path, img):
        cv2.imwrite(str(path), img)
        return f"/data/processed/{path.name}"

    def _augment(self, image, mode):
        h, w = image.shape[:2]
        if mode == "flip":
            return cv2.flip(image, 1)
        if mode == "rotate":
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
        if mode == "noise":
            n = np.random.normal(0, 15, image.shape).astype(np.int16)
            return np.clip(image.astype(np.int16) + n, 0, 255).astype(np.uint8)
        if mode == "brightness":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            scale = random.uniform(0.6, 1.4)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * scale, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if mode == "gan":
            shifted = np.clip(image.astype(np.int16) + np.random.randint(-20, 21, size=(1, 1, 3)), 0, 255).astype(np.uint8)
            out = shifted.copy(); ph, pw = max(8, h // 8), max(8, w // 8)
            for _ in range(8):
                y = random.randint(0, max(0, h - ph)); x = random.randint(0, max(0, w - pw))
                patch = out[y:y+ph, x:x+pw]
                out[y:y+ph, x:x+pw] = cv2.GaussianBlur(patch, (5, 5), 0)
            return out
        if mode == "crop":
            m_h, m_w = int(h * 0.1), int(w * 0.1)
            y1, y2 = m_h, h - m_h; x1, x2 = m_w, w - m_w
            crop = image[y1:y2, x1:x2]
            return cv2.resize(crop, (w, h))
        return image

    def process(self, file_path: Path, filename: str, job_id: str, augment: str = "none", pipeline: str = "full", task: str = "anomaly"):
        start = time.time()
        steps = []
        output_images = {"original": f"/data/uploads/{filename}"}
        try:
            img, media_type = self._extract_media_frame(file_path)
            if img is None:
                raise ValueError("Media rỗng")
            h0, w0 = img.shape[:2]
            steps.append({"name": "Tải media", "ok": True, "info": f"Media={media_type}, size={w0}x{h0}"})

            aug = self._augment(img, augment)
            p_aug = self.data_dir / "processed" / f"{job_id}_aug.jpg"
            output_images["augmented"] = self._save_img(p_aug, aug)
            steps.append({"name": "Tăng cường dữ liệu", "ok": True, "info": f"Chế độ={augment}", "img": output_images["augmented"]})

            h, w = aug.shape[:2]; scale = min(1.0, 640.0 / max(h, w))
            rs = cv2.resize(aug, (int(w * scale), int(h * scale)))
            lab = cv2.cvtColor(rs, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
            blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
            pre1 = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
            p_pre1 = self.data_dir / "processed" / f"{job_id}_pre1.jpg"
            output_images["pre1"] = self._save_img(p_pre1, pre1)
            steps.append({"name": "Tiền xử lý (1)", "ok": True, "info": "Resize<=640 + CLAHE + Unsharp", "img": output_images["pre1"]})

            bil = cv2.bilateralFilter(pre1, d=9, sigmaColor=75, sigmaSpace=75)
            gamma = 1.1
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
            gc = cv2.LUT(bil, lut)
            pre2 = cv2.medianBlur(gc, 3)
            p_pre2 = self.data_dir / "processed" / f"{job_id}_pre2.jpg"
            output_images["pre2"] = self._save_img(p_pre2, pre2)
            steps.append({"name": "Tiền xử lý (2)", "ok": True, "info": "Bilateral + Gamma + Median", "img": output_images["pre2"]})

            gray = cv2.cvtColor(pre2, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(edges, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_th = pre2.shape[0] * pre2.shape[1] * 0.003
            person_count = 0
            seg = pre2.copy(); ov = seg.copy()
            for c in cnts:
                area = cv2.contourArea(c)
                if area <= area_th: continue
                x,y,wc,hc = cv2.boundingRect(c)
                ratio = wc / max(hc, 1)
                if ratio > 0.8:
                    person_count += 1
                    cv2.rectangle(seg, (x,y), (x+wc, y+hc), (0,255,0), 2)
                    cv2.rectangle(ov, (x,y), (x+wc, y+hc), (0,255,0), -1)
            seg = cv2.addWeighted(seg, 0.8, ov, 0.2, 0)
            p_seg = self.data_dir / "processed" / f"{job_id}_seg.jpg"
            output_images["segmented"] = self._save_img(p_seg, seg)
            steps.append({"name": "Phân đoạn & phát hiện đối tượng", "ok": True, "info": f"Contours={len(cnts)}, person_count={person_count}", "img": output_images["segmented"]})

            sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            grad = cv2.magnitude(sobx, soby)
            density_map = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX)
            rnd = np.random.default_rng(seed=42)
            H, W = gray.shape
            sigma = max(H, W) / 15
            yy, xx = np.mgrid[0:H, 0:W]
            for _ in range(max(person_count, 1)):
                cx, cy = rnd.integers(0, W), rnd.integers(0, H)
                density_map += np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma*sigma))
            density_map = cv2.normalize(density_map, None, 0, 1, cv2.NORM_MINMAX)
            heat = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            density_vis = cv2.addWeighted(pre2, 0.5, heat, 0.5, 0)
            density_score = float(density_map.mean())
            density_level = "Thấp" if density_score < 0.2 else ("Vừa" if density_score < 0.5 else "Cao")
            p_den = self.data_dir / "processed" / f"{job_id}_density.jpg"
            output_images["density"] = self._save_img(p_den, density_vis)
            steps.append({"name": "Bản đồ mật độ", "ok": True, "info": f"density_score={density_score:.3f}, level={density_level}", "img": output_images["density"]})

            grayf = gray.astype(np.float32)
            sx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0)
            sy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(sx, sy, angleInDegrees=True)
            hsv = np.zeros((H, W, 3), dtype=np.uint8)
            hsv[..., 0] = (ang / 2).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_vis = cv2.addWeighted(pre2, 0.4, flow, 0.6, 0)
            stepv = max(H, W) // 20
            for y in range(stepv//2, H, stepv):
                for x in range(stepv//2, W, stepv):
                    dx = int(np.clip(sx[y, x], -20, 20)); dy = int(np.clip(sy[y, x], -20, 20))
                    if abs(dx)+abs(dy) > 2:
                        cv2.arrowedLine(flow_vis, (x, y), (x + dx, y + dy), (255, 255, 255), 1, tipLength=0.3)
            p_flow = self.data_dir / "processed" / f"{job_id}_flow.jpg"
            output_images["flow"] = self._save_img(p_flow, flow_vis)
            steps.append({"name": "Phân tích chuyển động", "ok": True, "info": "Sobel + HSV flow + vectors", "img": output_images["flow"]})

            texture_score = min(float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 2000, 1.0)
            motion_score = min(float(np.std(mag)) / 100, 1.0)
            density_norm = min(density_score * 3.0, 1.0)
            raw_score = texture_score * 0.35 + motion_score * 0.40 + density_norm * 0.25
            anomaly_score = min(raw_score * 1.8, 1.0)

            zone_idx = 0 if density_level == "Thấp" else (1 if density_level == "Vừa" else 2)
            if self.ae is not None:
                clip = np.repeat(pre2[np.newaxis, ...], 16, axis=0)
                clip_raw = self.feature_extractor.extract_raw(clip)
                anomaly_score = float(self.ae.score(clip_raw.reshape(1, -1))[0])
                tau = self.ccg.threshold(zone_idx)
                model_mode = "trained"
            else:
                gate = {"Thấp": 0.2, "Vừa": 0.5, "Cao": 0.8}[density_level]
                tau = 0.5 + 0.3 * gate
                model_mode = "simulation"

            rh, rw = pre2.shape[:2]
            zh, zw = rh // 3, rw // 4
            result = pre2.copy()
            for r in range(3):
                for c in range(4):
                    y1, y2 = r * zh, (r + 1) * zh if r < 2 else rh
                    x1, x2 = c * zw, (c + 1) * zw if c < 3 else rw
                    z = gray[y1:y2, x1:x2]
                    zscore = min(float(cv2.Laplacian(z, cv2.CV_32F).var()) / 1500, 1.0)
                    alertz = zscore > tau
                    color = (0, 0, 255) if alertz else (0, 180, 0)
                    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            bar = np.zeros((50, rw, 3), dtype=np.uint8)
            if anomaly_score >= 0.75: bar[:] = (30, 30, 200); status = "CẢNH BÁO"
            elif anomaly_score >= 0.55: bar[:] = (0, 215, 255); status = "NGHI NGỜ"
            else: bar[:] = (0, 150, 0); status = "BÌNH THƯỜNG"
            cv2.putText(bar, status, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(bar, f"Score: {anomaly_score:.3f}", (rw - 220, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            result = np.vstack([bar, result])
            p_result = self.data_dir / "processed" / f"{job_id}_result.jpg"
            output_images["result"] = self._save_img(p_result, result)
            steps.append({"name": "Phân loại bất thường DCAD + CCG", "ok": True, "info": f"tau={tau:.3f}, score={anomaly_score:.3f}", "img": output_images["result"]})

            if anomaly_score < 0.30: aclass = "Bình thường"
            elif anomaly_score < 0.55: aclass = "Nghi ngờ nhẹ"
            elif anomaly_score < 0.75: aclass = "Bất thường vừa"
            else: aclass = "Bất thường nghiêm trọng"
            alert = anomaly_score > 0.65

            elapsed = round(time.time() - start, 3)
            result_json = {
                "job_id": job_id,
                "filename": filename,
                "timestamp": datetime.utcnow().isoformat(),
                "elapsed_sec": elapsed,
                "media_type": media_type,
                "pipeline": pipeline,
                "task": task,
                "steps": steps,
                "summary": {
                    "anomaly_score": round(float(anomaly_score), 3),
                    "anomaly_class": aclass,
                    "person_count": int(person_count),
                    "density_level": density_level,
                    "density_score": round(float(density_score), 3),
                    "alert": bool(alert),
                    "threshold_used": round(float(tau), 3),
                    "model_used": self.model_name,
                    "model_mode": model_mode,
                },
                "output_images": output_images,
            }
            out_json = self.data_dir / "results" / f"{job_id}.json"
            out_json.write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
            return result_json
        except Exception as e:
            raise RuntimeError(str(e))
