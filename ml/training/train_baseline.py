
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

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
    print("HERMES - BASELINE FRAUD DETECTION MODEL")
    print("=" * 60)

    print("\n[1] Loading training and validation data")

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train[FEATURES]
    y_train = train["isFraud"]

    X_val = validation[FEATURES]
    y_val = validation["isFraud"]

    print(f"Training records: {len(train):,}")
    print(f"Validation records: {len(validation):,}")

    print("\n[2] Creating preprocessing pipeline")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    print("\n[3] Initializing Logistic Regression")

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    print("\n[4] Training model")

    model.fit(X_train, y_train)

    print("Model training completed")

    print("\n[5] Generating fraud predictions")

    probabilities = model.predict_proba(X_val)[:, 1]

    # Initial baseline threshold; not optimized yet
    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\n[6] Model Evaluation")

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

    roc_auc = roc_auc_score(y_val, probabilities)

    pr_auc = average_precision_score(
        y_val,
        probabilities,
    )

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {pr_auc:.4f}")

    print("\n[7] Saving trained pipeline")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        MODEL_DIR / "hermes_baseline.joblib"
    )

    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("HERMES BASELINE TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
