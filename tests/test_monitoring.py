import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.monitoring.drift import (
    build_feature_drift_summary,
    categorical_distribution_drift,
    population_stability_index,
)
from ml.monitoring.performance import evaluate_model_performance, evaluate_time_windows
from ml.monitoring.reporting import write_monitoring_reports
from ml.monitoring.run_monitoring import run_monitoring


class SyntheticProbabilityModel:
    def predict_proba(self, data):
        scores = data["score"].to_numpy(dtype=float)
        return np.column_stack([1 - scores, scores])


@pytest.fixture
def labeled_data():
    return pd.DataFrame({
        "score": [0.1, 0.2, 0.8, 0.9],
        "TransactionAmt": [10, 20, 30, 40],
        "TransactionDT": [1, 2, 3, 4],
        "isFraud": [0, 0, 1, 1],
    })


def test_numerical_psi_is_near_zero_for_identical_distributions():
    values = pd.Series([1, 2, 3, 4, 5] * 20)

    assert population_stability_index(values, values) < 1e-8


def test_numerical_psi_detects_significant_shift():
    reference = pd.Series(np.linspace(0, 1, 100))
    observed = pd.Series(np.linspace(10, 11, 100))

    assert population_stability_index(reference, observed) >= 0.25


def test_categorical_drift_handles_unseen_categories():
    distance, unseen_rate = categorical_distribution_drift(
        pd.Series(["a", "a", "b", None]),
        pd.Series(["z", "z", "z", None]),
    )

    assert distance > 0
    assert unseen_rate == 0.75


def test_feature_summary_tracks_missing_value_changes():
    reference = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0], "category": ["a"] * 4})
    observed = pd.DataFrame({"amount": [1.0, np.nan, np.nan, np.nan], "category": ["a"] * 4})

    summary = build_feature_drift_summary(
        reference,
        observed,
        numerical_features=["amount"],
        categorical_features=["category"],
    )

    amount_row = summary.loc[summary["feature"] == "amount"].iloc[0]
    assert amount_row["missing_rate_delta"] == 0.75
    assert amount_row["missing_severity"] == "high"


def test_performance_metrics_use_fixed_threshold(labeled_data):
    metrics = evaluate_model_performance(
        SyntheticProbabilityModel(),
        labeled_data,
        feature_columns=["score", "TransactionAmt"],
        threshold=0.5,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_count"] == 0
    assert metrics["alert_rate"] == 0.5


def test_time_windows_handle_single_class_windows(labeled_data):
    windows = evaluate_time_windows(
        SyntheticProbabilityModel(),
        labeled_data,
        feature_columns=["score", "TransactionAmt"],
        threshold=0.5,
        windows=2,
    )

    assert len(windows) == 2
    assert windows["pr_auc"].isna().all()
    assert windows["roc_auc"].isna().all()


def test_report_generation_writes_aggregate_reports(tmp_path, labeled_data):
    drift = pd.DataFrame([{
        "feature": "score",
        "reference_missing_rate": 0.0,
        "observed_missing_rate": 0.0,
        "missing_rate_delta": 0.0,
        "missing_severity": "low",
        "severity": "low",
    }])
    paths = write_monitoring_reports(
        tmp_path,
        feature_drift=drift,
        amount_distribution=pd.DataFrame({
            "bin_start": [0],
            "bin_end": [1],
            "reference_count": [2],
            "observed_count": [2],
        }),
        performance={"precision": 1.0},
        time_windows=pd.DataFrame({"window": [1], "precision": [1.0]}),
        methodology={"reference_dataset": "train.csv"},
    )

    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["model_performance"].read_text(encoding="utf-8"))["precision"] == 1.0


def test_run_monitoring_reports_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required monitoring input"):
        run_monitoring(tmp_path)


def test_existing_xgboost_model_is_compatible_with_monitoring():
    project_root = Path(__file__).resolve().parents[1]
    model = joblib.load(project_root / "models" / "hermes_xgboost.joblib")
    validation = pd.read_csv(project_root / "data" / "processed" / "validation.csv", nrows=20)
    with (project_root / "configs" / "model_config.json").open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    metrics = evaluate_model_performance(
        model,
        validation,
        feature_columns=config["feature_columns"],
        threshold=config["decision_threshold"],
    )

    assert metrics["transaction_count"] == 20
    assert "pr_auc" in metrics
