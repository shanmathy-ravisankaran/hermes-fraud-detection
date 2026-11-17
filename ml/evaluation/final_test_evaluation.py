"""Read-only final evaluation of saved fraud models on the held-out test split."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_THRESHOLD = 0.85
FEATURE_COLUMNS = [
    "TransactionAmt",
    "TransactionAmtLog",
    "TransactionHour",
    "TransactionDay",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
]


def _downsample_curve(points: dict[str, list[float]], maximum_points: int = 250) -> dict[str, list[float]]:
    point_count = len(next(iter(points.values()), []))
    if point_count <= maximum_points:
        return points
    indices = np.linspace(0, point_count - 1, maximum_points, dtype=int)
    return {key: [values[index] for index in indices] for key, values in points.items()}


def calculate_final_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Calculate fixed-threshold metrics without making any threshold choice."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be one-dimensional arrays of equal length.")
    if len(labels) == 0:
        raise ValueError("The held-out test dataset contains no rows.")

    alerts = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, alerts, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(labels)) == 2
    fraud_count = int(labels.sum())
    return {
        "transaction_count": len(labels),
        "fraud_count": fraud_count,
        "fraud_prevalence": float(labels.mean()),
        "decision_threshold": float(threshold),
        "true_negative_count": int(tn),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
        "precision": float(precision_score(labels, alerts, zero_division=0)),
        "recall": float(recall_score(labels, alerts, zero_division=0)),
        "f1_score": float(f1_score(labels, alerts, zero_division=0)),
        "accuracy": float(accuracy_score(labels, alerts)),
        "roc_auc": float(roc_auc_score(labels, scores)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(labels, scores)) if has_both_classes else None,
        "alert_count": int(alerts.sum()),
        "alert_rate": float(alerts.mean()),
        "fraud_capture_rate": float(tp / fraud_count) if fraud_count else None,
        "precision_at_threshold": float(precision_score(labels, alerts, zero_division=0)),
    }


def calculate_curve_data(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """Return bounded aggregate ROC and PR curves, or empty curves for one-class data."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) != 2:
        return {"roc": {"false_positive_rate": [], "true_positive_rate": []}, "pr": {"recall": [], "precision": []}}

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "roc": _downsample_curve({
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
        }),
        "pr": _downsample_curve({"recall": recall.tolist(), "precision": precision.tolist()}),
    }


def load_xgboost_threshold(report_path: Path) -> float:
    """Read the full persisted validation-selected XGBoost threshold."""
    if not report_path.is_file():
        raise RuntimeError(f"Authoritative XGBoost validation metrics report not found: {report_path}")
    report = pd.read_csv(report_path)
    matches = report.loc[report["model"] == "XGBoost", "threshold"] if "model" in report else pd.Series(dtype=float)
    if matches.empty:
        raise RuntimeError("Authoritative XGBoost threshold is missing from validation metrics.")
    threshold = float(matches.iloc[0])
    if not 0 < threshold < 1:
        raise RuntimeError("Authoritative XGBoost threshold must be between zero and one.")
    return threshold


def _validate_test_data(test_data: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + ["isFraud"])
    missing_columns = required_columns - set(test_data.columns)
    if missing_columns:
        raise ValueError(f"Held-out test dataset is missing columns: {sorted(missing_columns)}")


def _validate_model_features(model: Any, artifact_name: str) -> None:
    fitted_features = list(getattr(model, "feature_names_in_", []))
    if fitted_features != FEATURE_COLUMNS:
        raise ValueError(f"{artifact_name} does not expose the expected fitted feature columns.")


def write_final_evaluation_reports(
    output_directory: Path,
    *,
    model_metrics: dict[str, dict[str, Any]],
    curve_data: dict[str, dict[str, dict[str, list[float]]]],
    methodology: str,
) -> dict[str, Path]:
    """Write aggregate-only final test reports and bounded curve points."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_comparison": output_directory / "model_comparison.csv",
        "confusion_matrices": output_directory / "confusion_matrices.csv",
        "metrics": output_directory / "final_test_metrics.json",
        "curves": output_directory / "curve_data.json",
        "methodology": output_directory / "evaluation_methodology.md",
    }
    comparison_rows = []
    confusion_rows = []
    for model_name, metrics in model_metrics.items():
        comparison_rows.append({
            "model": model_name,
            **{key: value for key, value in metrics.items() if key not in {
                "true_negative_count", "false_positive_count", "false_negative_count", "true_positive_count"
            }},
        })
        confusion_rows.append({
            "model": model_name,
            "true_negative_count": metrics["true_negative_count"],
            "false_positive_count": metrics["false_positive_count"],
            "false_negative_count": metrics["false_negative_count"],
            "true_positive_count": metrics["true_positive_count"],
        })

    pd.DataFrame(comparison_rows).to_csv(paths["model_comparison"], index=False)
    pd.DataFrame(confusion_rows).to_csv(paths["confusion_matrices"], index=False)
    paths["metrics"].write_text(json.dumps(model_metrics, indent=2, sort_keys=True), encoding="utf-8")
    paths["curves"].write_text(json.dumps(curve_data, indent=2), encoding="utf-8")
    paths["methodology"].write_text(methodology, encoding="utf-8")
    return paths


def run_final_test_evaluation(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Evaluate saved baseline and XGBoost artifacts against test.csv exactly once per run."""
    test_path = project_root / "data" / "processed" / "test.csv"
    baseline_path = project_root / "models" / "hermes_baseline.joblib"
    xgboost_path = project_root / "models" / "hermes_xgboost.joblib"
    threshold_report_path = project_root / "docs" / "xgboost_validation_metrics.csv"
    for required_path in [test_path, baseline_path, xgboost_path]:
        if not required_path.is_file():
            raise FileNotFoundError(f"Required final evaluation input not found: {required_path}")

    xgboost_threshold = load_xgboost_threshold(threshold_report_path)
    test_data = pd.read_csv(test_path)
    _validate_test_data(test_data)
    labels = test_data["isFraud"].astype(int).to_numpy()
    models = {
        "Logistic Regression": (baseline_path, BASELINE_THRESHOLD),
        "XGBoost": (xgboost_path, xgboost_threshold),
    }
    model_metrics = {}
    curve_data = {}
    for model_name, (artifact_path, threshold) in models.items():
        model = joblib.load(artifact_path)
        _validate_model_features(model, artifact_path.name)
        scores = model.predict_proba(test_data[FEATURE_COLUMNS])[:, 1]
        metrics = calculate_final_metrics(labels, scores, threshold=threshold)
        metrics.update({
            "artifact_name": artifact_path.name,
            "artifact_version": "1.0.0" if model_name == "XGBoost" else "baseline",
            "dataset_split": "chronological held-out test.csv",
            "scores_calibrated": False,
        })
        model_metrics[model_name] = metrics
        curve_data[model_name] = calculate_curve_data(labels, scores)

    evaluation_date = datetime.now(UTC).isoformat()
    methodology = f"""# Hermes Final Held-Out Test Evaluation

Evaluation date: {evaluation_date}

## Scope

This report evaluates the immutable chronological `data/processed/test.csv` split only. No
model was retrained, refit, calibrated, tuned, or selected with test labels. The Logistic
Regression baseline uses its validation-selected threshold of {BASELINE_THRESHOLD:.2f}. XGBoost
uses the full persisted validation threshold of {xgboost_threshold:.16f}, loaded from
`docs/xgboost_validation_metrics.csv`.

## Metrics

Precision is TP/(TP+FP), recall/fraud capture is TP/(TP+FN), F1 is the harmonic mean of
precision and recall, and alert rate is alerts divided by all test transactions. ROC-AUC and
PR-AUC use score ranking only; operating metrics use the fixed thresholds above.

## Limitations

This is a research/demo evaluation, not production performance. Scores are uncalibrated model
outputs and are not fraud probabilities or confirmed fraud outcomes. Curves contain bounded,
aggregate points only; no transaction-level test data is exported.
"""
    report_paths = write_final_evaluation_reports(
        project_root / "docs" / "final_evaluation",
        model_metrics=model_metrics,
        curve_data=curve_data,
        methodology=methodology,
    )
    return {"metrics": model_metrics, "report_paths": report_paths, "xgboost_threshold": xgboost_threshold}


def main() -> None:
    results = run_final_test_evaluation()
    print("Hermes final held-out test evaluation complete")
    print(f"XGBoost fixed threshold: {results['xgboost_threshold']:.16f}")
    for model_name, metrics in results["metrics"].items():
        print(
            f"{model_name}: PR-AUC {metrics['pr_auc']:.4f} | "
            f"F1 {metrics['f1_score']:.4f} | alerts {metrics['alert_count']:,}"
        )
    print("Reports:")
    for report_name, report_path in results["report_paths"].items():
        print(f"  {report_name}: {report_path}")


if __name__ == "__main__":
    main()
