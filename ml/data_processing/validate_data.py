
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"


def validate_transactions():
    print("\n" + "=" * 60)
    print("HERMES - TRANSACTION DATA QUALITY VALIDATION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {DATA_PATH}"
        )

    required_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
    ]

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"\nTotal records: {len(df):,}")

    # 1. Validate required columns
    print("\n[1] Required Column Validation")

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("PASS - All required columns are present")

    # 2. Duplicate transaction IDs
    print("\n[2] Duplicate Transaction IDs")

    duplicate_count = df["TransactionID"].duplicated().sum()

    print(f"Duplicate IDs: {duplicate_count:,}")

    # 3. Missing fraud labels
    print("\n[3] Missing Fraud Labels")

    missing_labels = df["isFraud"].isna().sum()

    print(f"Missing labels: {missing_labels:,}")

    # 4. Fraud label validity
    print("\n[4] Fraud Label Validation")

    invalid_labels = (
        ~df["isFraud"].isin([0, 1])
    ).sum()

    print(f"Invalid labels: {invalid_labels:,}")

    # 5. Transaction amount validation
    print("\n[5] Transaction Amount Validation")

    invalid_amounts = (
        df["TransactionAmt"].isna()
        | (df["TransactionAmt"] <= 0)
    ).sum()

    print(f"Missing or non-positive amounts: {invalid_amounts:,}")

    # 6. Missing value analysis
    print("\n[6] Missing Value Summary")

    missing_percentages = (
        df.isna().mean() * 100
    )

    high_missing_columns = (
        missing_percentages > 80
    ).sum()

    print(
        f"Columns with over 80% missing values: "
        f"{high_missing_columns}"
    )

    # 7. Transaction timestamp validation
    print("\n[7] Transaction Time Validation")

    missing_times = df["TransactionDT"].isna().sum()
    negative_times = (df["TransactionDT"] < 0).sum()
    ordered = df["TransactionDT"].is_monotonic_increasing

    print(f"Missing timestamps: {missing_times:,}")
    print(f"Negative timestamps: {negative_times:,}")
    print(f"Transactions ordered by time: {ordered}")

    # Final quality summary
    print("\n" + "=" * 60)

    critical_issues = (
        duplicate_count
        + missing_labels
        + invalid_labels
        + invalid_amounts
        + missing_times
        + negative_times
    )

    if critical_issues == 0:
        print("RESULT: CORE DATA VALIDATION PASSED")
    else:
        print("RESULT: DATA QUALITY ISSUES DETECTED")
        print(f"Total issue occurrences: {critical_issues:,}")

    print("=" * 60)


if __name__ == "__main__":
    validate_transactions()
