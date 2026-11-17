
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "docs"

NUMERIC_FEATURES = [
    "TransactionAmt",
    "TransactionAmtLog",
    "TransactionHour",
    "TransactionDay",
]

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    print("\n" + "=" * 60)
    print("HERMES - XGBOOST FRAUD DETECTION")
    print("=" * 60)

    print("\n[1] Loading training and validation datasets")

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train[FEATURES]
    y_train = train["isFraud"]

    X_val = val[FEATURES]
    y_val = val["isFraud"]

    print(f"Training transactions: {len(train):,}")
    print(f"Validation transactions: {len(val):,}")

    print("\n[2] Configuring preprocessing")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    fraud_count = int(y_train.sum())
    legitimate_count = int(len(y_train) - fraud_count)

    scale_pos_weight = legitimate_count / fraud_count

    print(f"Class imbalance weight: {scale_pos_weight:.2f}")

    print("\n[3] Initializing XGBoost")

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    print("\n[4] Training XGBoost")

    model.fit(X_train, y_train)

    print("XGBoost training completed")

    print("\n[5] Evaluating validation predictions")

    scores = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, scores)
    pr_auc = average_precision_score(y_val, scores)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {pr_auc:.4f}")

    print("\n[6] Finding best F1 threshold")

    precision, recall, thresholds = precision_recall_curve(
        y_val, scores
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    predictions = (
        scores >= best_threshold
    ).astype(int)

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best F1-score: {f1_scores[best_idx]:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("\n[7] Saving model and validation metrics")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "hermes_xgboost.joblib",
    )

    report = pd.DataFrame([{
        "model": "XGBoost",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": float(f1_scores[best_idx]),
        "threshold": best_threshold,
    }])

    report.to_csv(
        REPORT_DIR / "xgboost_validation_metrics.csv",
        index=False,
    )

    print("Model and metrics saved successfully")

    print("\n" + "=" * 60)
    print("HERMES XGBOOST TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
