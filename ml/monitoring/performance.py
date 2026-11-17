"""Historical offline model-performance monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model_performance(
    model: Any,
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    threshold: float,
    label_column: str = "isFraud",
) -> dict[str, Any]:
    """Evaluate a fixed model and fixed decision threshold on labeled historical data."""
    if label_column not in dataset.columns:
        raise ValueError(f"Missing label column: {label_column}")
    missing_features = set(feature_columns) - set(dataset.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")

    labels = dataset[label_column].astype(int).to_numpy()
    scores = model.predict_proba(dataset[feature_columns])[:, 1]
    alerts = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, alerts, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(labels)) == 2
    fraud_count = int(labels.sum())

    return {
        "transaction_count": len(dataset),
        "fraud_count": fraud_count,
        "observed_fraud_rate": float(labels.mean()) if len(labels) else 0.0,
        "precision": float(precision_score(labels, alerts, zero_division=0)),
        "recall": float(recall_score(labels, alerts, zero_division=0)),
        "f1_score": float(f1_score(labels, alerts, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(labels, scores)) if has_both_classes else None,
        "true_negative_count": int(tn),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
        "alert_rate": float(alerts.mean()) if len(alerts) else 0.0,
        "fraud_capture_rate": float(tp / fraud_count) if fraud_count else None,
        "decision_threshold": float(threshold),
    }


def evaluate_time_windows(
    model: Any,
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    threshold: float,
    windows: int = 6,
    time_column: str = "TransactionDT",
) -> pd.DataFrame:
    """Evaluate sequential chronological validation windows without using labels as inputs."""
    if time_column not in dataset.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if windows < 1:
        raise ValueError("windows must be at least 1")

    chronological_data = dataset.sort_values(time_column).reset_index(drop=True)
    rows = []
    window_indices = np.array_split(np.arange(len(chronological_data)), windows)
    for window_number, indices in enumerate(window_indices, start=1):
        window_data = chronological_data.iloc[indices]
        if window_data.empty:
            continue
        metrics = evaluate_model_performance(
            model,
            window_data,
            feature_columns=feature_columns,
            threshold=threshold,
        )
        rows.append({
            "window": window_number,
            "period_start": int(window_data[time_column].min()),
            "period_end": int(window_data[time_column].max()),
            "mean_transaction_amount": float(window_data["TransactionAmt"].mean()),
            **metrics,
        })

    return pd.DataFrame(rows)
