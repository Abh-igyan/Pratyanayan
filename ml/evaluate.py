from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred_proba)
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
    calibration = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "brier_score": float(brier),
        "calibration": calibration,
        "predicted_probability_distribution": {
            "min": float(np.min(y_score)),
            "p25": float(np.quantile(y_score, 0.25)),
            "median": float(np.median(y_score)),
            "p75": float(np.quantile(y_score, 0.75)),
            "max": float(np.max(y_score)),
            "mean": float(np.mean(y_score)),
        },
        "precision_recall_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
        },
        "threshold": threshold,
    }


def business_metrics(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    selected_mask = y_proba >= threshold
    selected = df.loc[selected_mask].copy()
    recovered_orders = int(selected["recovery_success"].sum())
    recovered_revenue = float(selected.loc[selected["recovery_success"] == 1, "amount"].sum())
    intervention_revenue = float(selected["amount"].sum())
    recovery_attempts_selected = int(len(selected))
    intervention_rate = float(recovery_attempts_selected / len(df)) if len(df) else 0.0
    revenue_recovery_rate = recovered_revenue / intervention_revenue if intervention_revenue else 0.0
    return {
        "recovery_attempts_selected": recovery_attempts_selected,
        "recovered_orders": recovered_orders,
        "recovered_revenue": recovered_revenue,
        "revenue_recovery_rate": float(revenue_recovery_rate),
        "intervention_rate": float(intervention_rate),
    }
