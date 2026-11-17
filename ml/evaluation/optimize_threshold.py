
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hermes_baseline.joblib"

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation.csv"
)

REPORT_DIR = PROJECT_ROOT / "docs"

FEATURES = [
    "TransactionAmt",
    "TransactionAmtLog",
    "TransactionHour",
    "TransactionDay",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
]


def main():
    print("\n" + "=" * 65)
    print("HERMES - FRAUD THRESHOLD OPTIMIZATION")
    print("=" * 65)

    print("\n[1] Loading trained baseline model")

    model = joblib.load(MODEL_PATH)

    print("\n[2] Loading validation dataset")

    validation = pd.read_csv(VALIDATION_PATH)

    X_val = validation[FEATURES]
    y_val = validation["isFraud"]

    print(f"Validation records: {len(validation):,}")

    print("\n[3] Generating prediction scores")

    scores = model.predict_proba(X_val)[:, 1]

    thresholds = np.round(
        np.arange(0.10, 0.96, 0.05),
        2,
    )

    results = []

    print("\n[4] Evaluating thresholds")

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        _tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "alerts": int(tp + fp),
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(index=False)
    )

    print("\n[5] Selecting best F1 threshold")

    best = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1-score: {best['f1']:.4f}")
    print(f"False positives: {int(best['false_positives']):,}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "threshold_comparison.csv"

    results_df.to_csv(report_path, index=False)

    print(f"\nReport saved: {report_path}")

    print("\n" + "=" * 65)
    print("HERMES THRESHOLD ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
