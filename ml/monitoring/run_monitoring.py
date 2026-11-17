"""CLI entry point for Hermes historical offline monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.monitoring.drift import (
    CATEGORICAL_TVD_THRESHOLDS,
    MISSING_RATE_THRESHOLDS,
    NUMERICAL_PSI_THRESHOLDS,
    build_feature_drift_summary,
    transaction_amount_distribution,
)
from ml.monitoring.performance import evaluate_model_performance, evaluate_time_windows
from ml.monitoring.reporting import write_monitoring_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUMERICAL_FEATURES = [
    "TransactionAmt",
    "TransactionAmtLog",
    "TransactionHour",
    "TransactionDay",
]
CATEGORICAL_FEATURES = ["ProductCD", "card4", "card6", "P_emaildomain"]


def run_monitoring(project_root: Path = PROJECT_ROOT) -> dict:
    """Generate offline validation monitoring reports from preserved train/validation splits."""
    config_path = project_root / "configs" / "model_config.json"
    train_path = project_root / "data" / "processed" / "train.csv"
    validation_path = project_root / "data" / "processed" / "validation.csv"
    report_directory = project_root / "docs" / "monitoring"

    for path in [config_path, train_path, validation_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required monitoring input not found: {path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    model_path = project_root / "models" / config["model_file"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    model = joblib.load(model_path)
    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(validation_path)
    feature_columns = config["feature_columns"]
    threshold = float(config["decision_threshold"])

    feature_drift = build_feature_drift_summary(
        train_data,
        validation_data,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    amount_distribution = transaction_amount_distribution(
        train_data["TransactionAmt"],
        validation_data["TransactionAmt"],
    )
    performance = evaluate_model_performance(
        model,
        validation_data,
        feature_columns=feature_columns,
        threshold=threshold,
    )
    performance.update({
        "dataset": "validation.csv",
        "evaluation_period": "chronological validation split",
        "model_name": config["model_name"],
        "score_calibrated": bool(config["score_calibrated"]),
    })
    time_windows = evaluate_time_windows(
        model,
        validation_data,
        feature_columns=feature_columns,
        threshold=threshold,
    )
    methodology = {
        "reference_dataset": "train.csv",
        "observed_dataset": "validation.csv",
        "decision_threshold": threshold,
        "time_windows": 6,
        "numerical_drift": {
            "metric": "population stability index",
            "thresholds": NUMERICAL_PSI_THRESHOLDS,
        },
        "categorical_drift": {
            "metric": "total variation distance",
            "thresholds": CATEGORICAL_TVD_THRESHOLDS,
        },
        "missing_value_drift": {
            "metric": "absolute missing-rate change",
            "thresholds": MISSING_RATE_THRESHOLDS,
        },
        "limitations": [
            "Thresholds are heuristic monitoring indicators, not statistically validated alerts.",
            "Results are historical offline validation monitoring, not production performance.",
            "Model scores are uncalibrated and are not fraud probabilities.",
            "The held-out test split is not used by this monitoring command.",
        ],
    }
    paths = write_monitoring_reports(
        report_directory,
        feature_drift=feature_drift,
        amount_distribution=amount_distribution,
        performance=performance,
        time_windows=time_windows,
        methodology=methodology,
    )
    return {
        "performance": performance,
        "feature_drift": feature_drift,
        "time_windows": time_windows,
        "report_paths": paths,
    }


def main() -> None:
    results = run_monitoring()
    performance = results["performance"]
    severity_counts = results["feature_drift"]["severity"].value_counts().to_dict()
    print("Hermes historical offline monitoring complete")
    print(f"Dataset: {performance['dataset']} ({performance['evaluation_period']})")
    print(f"Transactions: {performance['transaction_count']:,}")
    print(f"PR-AUC: {performance['pr_auc']:.4f}" if performance["pr_auc"] is not None else "PR-AUC: n/a")
    print(f"Alert rate: {performance['alert_rate']:.2%}")
    print(f"Feature drift severities: {severity_counts}")
    print("Reports:")
    for name, path in results["report_paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
