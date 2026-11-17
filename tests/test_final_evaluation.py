import hashlib

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.final_test_evaluation import (
    FEATURE_COLUMNS,
    calculate_curve_data,
    calculate_final_metrics,
    run_final_test_evaluation,
    write_final_evaluation_reports,
)


class NoFitModel:
    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def fit(self, *args, **kwargs):
        raise AssertionError("Final evaluation must not train or refit models")

    def predict_proba(self, data):
        scores = data["TransactionAmt"].to_numpy(dtype=float) / 100
        return np.column_stack([1 - scores, scores])


def test_fixed_threshold_metrics_and_confusion_matrix_order():
    metrics = calculate_final_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.9, 0.2, 0.9]),
        threshold=0.85,
    )

    assert metrics["true_negative_count"] == 1
    assert metrics["false_positive_count"] == 1
    assert metrics["false_negative_count"] == 1
    assert metrics["true_positive_count"] == 1
    assert metrics["alert_count"] == 2


def test_threshold_includes_scores_equal_to_operating_threshold():
    metrics = calculate_final_metrics(
        np.array([0, 1]),
        np.array([0.84, 0.85]),
        threshold=0.85,
    )

    assert metrics["alert_count"] == 1
    assert metrics["true_positive_count"] == 1


def test_one_class_edge_case_has_safe_auc_values():
    metrics = calculate_final_metrics(
        np.array([0, 0]),
        np.array([0.1, 0.2]),
        threshold=0.85,
    )
    curves = calculate_curve_data(np.array([0, 0]), np.array([0.1, 0.2]))

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert curves["roc"]["false_positive_rate"] == []


def test_final_report_generation(tmp_path):
    paths = write_final_evaluation_reports(
        tmp_path,
        model_metrics={
            "XGBoost": {
                "transaction_count": 2,
                "true_negative_count": 1,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "true_positive_count": 1,
            }
        },
        curve_data={"XGBoost": {"roc": {}, "pr": {}}},
        methodology="# Methodology\n",
    )

    assert all(path.is_file() for path in paths.values())
    assert "XGBoost" in paths["model_comparison"].read_text(encoding="utf-8")


def test_final_evaluation_uses_held_out_test_and_does_not_modify_model_artifacts(tmp_path):
    project_root = tmp_path
    (project_root / "data" / "processed").mkdir(parents=True)
    (project_root / "models").mkdir()
    (project_root / "docs").mkdir()
    test_data = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [1, 2],
        "isFraud": [0, 1],
        "TransactionAmt": [10.0, 90.0],
        "TransactionAmtLog": [1.0, 4.5],
        "TransactionHour": [0, 1],
        "TransactionDay": [0, 0],
        "ProductCD": ["W", "C"],
        "card4": ["visa", "visa"],
        "card6": ["debit", "credit"],
        "P_emaildomain": ["missing", "missing"],
    })
    test_data.to_csv(project_root / "data" / "processed" / "test.csv", index=False)
    baseline_path = project_root / "models" / "hermes_baseline.joblib"
    xgboost_path = project_root / "models" / "hermes_xgboost.joblib"
    joblib.dump(NoFitModel(), baseline_path)
    joblib.dump(NoFitModel(), xgboost_path)
    (project_root / "docs" / "xgboost_validation_metrics.csv").write_text(
        "model,threshold\nXGBoost,0.7593424320220947\n",
        encoding="utf-8",
    )
    before_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [baseline_path, xgboost_path]
    }

    result = run_final_test_evaluation(project_root)

    after_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [baseline_path, xgboost_path]
    }
    assert result["metrics"]["XGBoost"]["dataset_split"] == "chronological held-out test.csv"
    assert result["xgboost_threshold"] == 0.7593424320220947
    assert before_hashes == after_hashes
