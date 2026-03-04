"""
src/reporting/reporter.py — PDF & CSV Report Exporter
======================================================

Generates professional reports from pipeline run results:
  • CSV: raw anomaly event log (for spreadsheet analysis)
  • PDF: formatted summary with charts, timeline, stats
  • JSON: machine-readable full report

Usage:
    from src.reporting.reporter import Reporter
    reporter = Reporter(run_summary, output_dir="outputs/reports")
    reporter.export_csv("anomalies.csv")
    reporter.export_pdf("report.pdf")   # requires reportlab
    reporter.export_json("report.json")
"""
from __future__ import annotations
import csv
import json
import time
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────

class RunSummary:
    """
    Aggregates all data from a VideoProcessor run.
    Pass this to Reporter to generate output files.
    """

    def __init__(
        self,
        video_path: str,
        frames_processed: int,
        fps_achieved: float,
        elapsed_sec: float,
        anomaly_log: List[Dict],
        class_counts: Optional[Dict[str, int]] = None,
        config_snapshot: Optional[Dict] = None,
    ):
        self.video_path       = video_path
        self.frames_processed = frames_processed
        self.fps_achieved     = fps_achieved
        self.elapsed_sec      = elapsed_sec
        self.anomaly_log      = anomaly_log
        self.class_counts     = class_counts or {}
        self.config_snapshot  = config_snapshot or {}
        self.generated_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.total_anomalies  = len(anomaly_log)

    def to_dict(self) -> Dict:
        return {
            "generated_at":     self.generated_at,
            "video_path":       self.video_path,
            "frames_processed": self.frames_processed,
            "fps_achieved":     self.fps_achieved,
            "elapsed_sec":      self.elapsed_sec,
            "total_anomalies":  self.total_anomalies,
            "class_counts":     self.class_counts,
            "anomaly_log":      self.anomaly_log,
        }


# ── Reporter ──────────────────────────────────────────────────────────────

class Reporter:
    """
    Export run results to CSV, PDF, and JSON.

    Args:
        summary:    RunSummary object from a completed pipeline run
        output_dir: directory to write report files
    """

    def __init__(self, summary: RunSummary, output_dir: str = "outputs/reports"):
        self.summary    = summary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────

    def export_csv(self, filename: str = "anomaly_log.csv") -> Path:
        """
        Export anomaly event log as CSV.

        Columns: time, frame, track_id, anomaly_type, center_x, center_y
        """
        out_path = self.output_dir / filename
        fields = ["time", "frame", "track_id", "type", "center_x", "center_y"]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for event in self.summary.anomaly_log:
                center = event.get("center", (0, 0))
                writer.writerow({
                    "time":       event.get("time", ""),
                    "frame":      event.get("frame", ""),
                    "track_id":   event.get("track_id", ""),
                    "type":       event.get("type", ""),
                    "center_x":   center[0] if isinstance(center, (list, tuple)) else 0,
                    "center_y":   center[1] if isinstance(center, (list, tuple)) else 0,
                })

        logger.info(f"CSV exported → {out_path}  ({len(self.summary.anomaly_log)} rows)")
        return out_path

    def export_stats_csv(self, filename: str = "run_stats.csv") -> Path:
        """Export per-minute anomaly counts for time-series analysis."""
        out_path = self.output_dir / filename

        # Bucket anomalies by minute
        buckets: Dict[str, int] = {}
        for event in self.summary.anomaly_log:
            minute = event.get("time", "00:00:00")[:5]   # HH:MM
            buckets[minute] = buckets.get(minute, 0) + 1

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["minute", "anomaly_count"])
            for minute in sorted(buckets):
                writer.writerow([minute, buckets[minute]])

        logger.info(f"Stats CSV exported → {out_path}")
        return out_path

    # ── JSON ──────────────────────────────────────────────────────────────

    def export_json(self, filename: str = "report.json") -> Path:
        """Export full run summary as JSON."""
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.summary.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"JSON exported → {out_path}")
        return out_path

    # ── PDF ───────────────────────────────────────────────────────────────

    def export_pdf(self, filename: str = "report.pdf") -> Path:
        """
        Generate a formatted PDF report.
        Uses reportlab if available, falls back to plain text .txt file.
        """
        out_path = self.output_dir / filename
        try:
            self._export_pdf_reportlab(out_path)
        except ImportError:
            logger.warning("reportlab not installed → generating plain text report instead")
            txt_path = out_path.with_suffix(".txt")
            self._export_txt(txt_path)
            return txt_path
        return out_path

    def _export_pdf_reportlab(self, out_path: Path):
        """Full PDF using reportlab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable
        )

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        styles = getSampleStyleSheet()
        s = self.summary

        # ── Custom styles ────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#0D1B2A"),
            spaceAfter=6,
        )
        h2_style = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#0D9488"),
            spaceBefore=14, spaceAfter=4,
        )
        normal = styles["Normal"]
        normal.fontSize = 10

        elements = []

        # ── Title ─────────────────────────────────────────────────────────
        elements.append(Paragraph(
            "Vision-Based Phát Hiện Hành Vi Bất Thường", title_style
        ))
        elements.append(Paragraph(
            f"Báo Cáo Phân Tích — {s.generated_at}", normal
        ))
        elements.append(HRFlowable(width="100%", thickness=2,
                                    color=colors.HexColor("#0D9488")))
        elements.append(Spacer(1, 0.4*cm))

        # ── Run overview table ────────────────────────────────────────────
        elements.append(Paragraph("1. Tổng Quan", h2_style))
        overview_data = [
            ["Thông số", "Giá trị"],
            ["Video đầu vào",    str(Path(s.video_path).name)],
            ["Frames xử lý",    str(s.frames_processed)],
            ["FPS đạt được",    f"{s.fps_achieved:.1f}"],
            ["Thời gian chạy",  f"{s.elapsed_sec:.1f}s"],
            ["Tổng cảnh báo",   str(s.total_anomalies)],
            ["Thời điểm tạo",   s.generated_at],
        ]
        tbl = Table(overview_data, colWidths=[6*cm, 10*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#0D1B2A")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTSIZE",    (0,0), (-1,0), 11),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND",  (0,1), (0,-1), colors.HexColor("#F1F5F9")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ("FONTSIZE",    (0,1), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING",  (0,0), (-1,-1), 5),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 0.5*cm))

        # ── Anomaly breakdown table ───────────────────────────────────────
        from collections import Counter
        type_counts = Counter(e.get("type", "Unknown") for e in s.anomaly_log)

        elements.append(Paragraph("2. Phân Tích Cảnh Báo", h2_style))
        if type_counts:
            anom_data = [["Loại bất thường", "Số lượng", "Tỷ lệ"]]
            total = sum(type_counts.values())
            for anom_type, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                pct = f"{cnt/total*100:.1f}%"
                anom_data.append([anom_type, str(cnt), pct])
            anom_data.append(["TỔNG CỘNG", str(total), "100%"])

            anom_tbl = Table(anom_data, colWidths=[8*cm, 4*cm, 4*cm])
            anom_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#0D9488")),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("BACKGROUND",  (0,-1), (-1,-1), colors.HexColor("#FEF3C7")),
                ("FONTNAME",    (0,-1), (-1,-1), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#F1F5F9")]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ("ALIGN",       (1,0), (-1,-1), "CENTER"),
                ("FONTSIZE",    (0,0), (-1,-1), 10),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
            ]))
            elements.append(anom_tbl)
        else:
            elements.append(Paragraph("Không phát hiện hành vi bất thường.", normal))
        elements.append(Spacer(1, 0.5*cm))

        # ── Event log (first 30) ──────────────────────────────────────────
        elements.append(Paragraph("3. Nhật Ký Sự Kiện (30 đầu tiên)", h2_style))
        if s.anomaly_log:
            log_data = [["Thời gian", "Frame", "Track ID", "Loại"]]
            for event in s.anomaly_log[:30]:
                log_data.append([
                    event.get("time", ""),
                    str(event.get("frame", "")),
                    str(event.get("track_id", "")),
                    event.get("type", ""),
                ])
            log_tbl = Table(log_data, colWidths=[3.5*cm, 3*cm, 3.5*cm, 6*cm])
            log_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#0D1B2A")),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#E2E8F0")),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING",  (0,0), (-1,-1), 4),
            ]))
            elements.append(log_tbl)
        else:
            elements.append(Paragraph("Không có sự kiện nào.", normal))

        elements.append(Spacer(1, 0.5*cm))
        elements.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor("#CBD5E1")))
        elements.append(Paragraph(
            f"Báo cáo tự động sinh bởi Vision-Based Detection System — {s.generated_at}",
            ParagraphStyle("footer", parent=normal, fontSize=8,
                           textColor=colors.HexColor("#94A3B8"))
        ))

        doc.build(elements)
        logger.info(f"PDF exported → {out_path}")

    def _export_txt(self, out_path: Path):
        """Plain text fallback when reportlab is unavailable."""
        s = self.summary
        lines = [
            "=" * 60,
            "  VISION-BASED BEHAVIOR DETECTION — BÁO CÁO",
            "=" * 60,
            f"  Thời điểm     : {s.generated_at}",
            f"  Video         : {s.video_path}",
            f"  Frames        : {s.frames_processed}",
            f"  FPS đạt       : {s.fps_achieved:.1f}",
            f"  Thời gian     : {s.elapsed_sec:.1f}s",
            f"  Tổng cảnh báo : {s.total_anomalies}",
            "",
            "  ANOMALY LOG:",
            "-" * 60,
        ]
        for ev in s.anomaly_log[:50]:
            lines.append(
                f"  [{ev.get('time','')}] F{ev.get('frame',0):05d} "
                f"Track#{ev.get('track_id','')}  {ev.get('type','')}"
            )
        lines += ["=" * 60, ""]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"TXT report exported → {out_path}")
