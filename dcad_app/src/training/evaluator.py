from __future__ import annotations

import numpy as np


class Evaluator:
    def compute_auc(self, scores, labels) -> float:
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        thresholds = np.unique(scores)[::-1]
        pos = max(labels.sum(), 1)
        neg = max((1 - labels).sum(), 1)
        tprs, fprs = [0.0], [0.0]
        for t in thresholds:
            pred = (scores >= t).astype(np.int32)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            tprs.append(tp / pos)
            fprs.append(fp / neg)
        tprs.append(1.0); fprs.append(1.0)
        order = np.argsort(fprs)
        return float(np.trapz(np.array(tprs)[order], np.array(fprs)[order]))

    def compute_eer(self, scores, labels) -> float:
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        pos = max(labels.sum(), 1)
        neg = max((1 - labels).sum(), 1)
        best_gap, best_eer = 1e9, 1.0
        for t in np.unique(scores):
            pred = (scores >= t).astype(int)
            fpr = ((pred == 1) & (labels == 0)).sum() / neg
            fnr = ((pred == 0) & (labels == 1)).sum() / pos
            gap = abs(fpr - fnr)
            if gap < best_gap:
                best_gap = gap
                best_eer = (fpr + fnr) / 2
        return float(best_eer)
