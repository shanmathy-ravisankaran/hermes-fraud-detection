
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features():
    print("\n" + "=" * 60)
    print("HERMES - FEATURE ENGINEERING")
    print("=" * 60)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(RAW_DATA_PATH)

    required_columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
    ]

    print("\n[1] Loading required transaction columns")

    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=required_columns,
    )

    # Keep chronological order
    df = df.sort_values(
        ["TransactionDT", "TransactionID"]
    ).reset_index(drop=True)

    print(f"Loaded {len(df):,} transactions")

    print("\n[2] Creating numerical features")

    df["TransactionAmtLog"] = np.log1p(
        df["TransactionAmt"]
    )

    df["TransactionHour"] = (
        (df["TransactionDT"] // 3600) % 24
    ).astype("int16")

    df["TransactionDay"] = (
        df["TransactionDT"] // 86400
    ).astype("int32")

    print("\n[3] Preparing categorical features")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string")
        df[column] = df[column].fillna("missing")
        df[column] = df[column].astype(str)

    print("\n[4] Creating chronological splits")

    total = len(df)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[5] Saving datasets")

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    output_columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
    ] + FEATURE_COLUMNS

    for name, dataset in splits.items():
        output_path = (
            PROCESSED_DIR / f"{name}.csv"
        )

        dataset[output_columns].to_csv(
            output_path,
            index=False,
        )

        fraud_rate = (
            dataset["isFraud"].mean() * 100
        )

        print(
            f"{name}: {len(dataset):,} records "
            f"| fraud rate: {fraud_rate:.2f}%"
        )

    print("\n[6] Checking split boundaries")

    assert (
        train["TransactionDT"].max()
        <= validation["TransactionDT"].min()
    )

    assert (
        validation["TransactionDT"].max()
        <= test["TransactionDT"].min()
    )

    print("Chronological split checks passed")

    print("\n" + "=" * 60)
    print("HERMES FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    build_features()
