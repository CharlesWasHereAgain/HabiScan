"""Evaluation metrics used in Phase 3 and later phases."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def image_metrics(y_true, scores, threshold: float) -> dict:
    """Detection metrics: AUROC (threshold-free) + precision/recall/F1 at our threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, float)
    pred = (s > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"auroc": float(roc_auc_score(y, s)), "avg_precision": float(average_precision_score(y, s)),
            "threshold": float(threshold), "precision": prec, "recall": rec, "f1": f1,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_points(y_true, scores):
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(scores, float))
    return fpr, tpr


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, gt).sum()
    a, b = pred.sum(), gt.sum()
    dice = 2 * inter / (a + b) if a + b else 1.0
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    return float(dice), float(iou)


def pointing_hit(a: np.ndarray, gt: np.ndarray, tolerance_px: int = 16) -> bool:
    """Does the single most anomalous pixel fall on (or within tolerance of) the defect?"""
    y, x = np.unravel_index(np.argmax(a), a.shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    near = cv2.dilate(gt.astype(np.uint8), k) > 0
    return bool(near[y, x])


def pixel_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels.astype(int), scores))
