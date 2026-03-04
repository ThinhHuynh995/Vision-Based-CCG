from __future__ import annotations

import json
from pathlib import Path


def get_stats(data_dir: Path) -> dict:
    results_dir = data_dir / "results"
    uploads_dir = data_dir / "uploads"
    stats = {"total_processed": 0, "total_uploads": 0, "alerts": 0, "normal": 0}
    if uploads_dir.exists():
        stats["total_uploads"] = sum(1 for p in uploads_dir.iterdir() if p.is_file())
    if not results_dir.exists():
        return stats
    for file in results_dir.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            stats["total_processed"] += 1
            if data.get("summary", {}).get("alert"):
                stats["alerts"] += 1
            else:
                stats["normal"] += 1
        except Exception:
            continue
    return stats


def list_history(results_dir: Path, limit: int = 20) -> list[dict]:
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]
    items = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            s = data.get("summary", {})
            items.append(
                {
                    "job_id": data.get("job_id"),
                    "filename": data.get("filename"),
                    "timestamp": data.get("timestamp"),
                    "anomaly_score": s.get("anomaly_score", 0),
                    "anomaly_class": s.get("anomaly_class", "N/A"),
                    "alert": s.get("alert", False),
                    "person_count": s.get("person_count", 0),
                    "density_level": s.get("density_level", "N/A"),
                    "result_img": data.get("output_images", {}).get("result"),
                    "elapsed_sec": data.get("elapsed_sec", 0),
                }
            )
        except Exception:
            continue
    return items
