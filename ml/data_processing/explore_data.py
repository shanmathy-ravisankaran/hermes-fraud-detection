
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"


def explore_transactions():
    print("\n" + "=" * 55)
    print("HERMES - FRAUD DETECTION DATA ANALYSIS")
    print("=" * 55)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load transaction data
    print("\n[1] Loading transaction dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"Total transactions: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Fraud distribution
    print("\n[2] Fraud Distribution")

    fraud_counts = df["isFraud"].value_counts()
    fraud_percentages = (
        df["isFraud"].value_counts(normalize=True) * 100
    )

    print("Transaction counts:")
    print(fraud_counts)

    print("\nTransaction percentages:")
    print(fraud_percentages.round(2))

    # Transaction amount analysis
    print("\n[3] Transaction Amount Statistics")

    print(df["TransactionAmt"].describe())

    # Missing values
    print("\n[4] Top 10 Columns With Missing Values")

    missing_percentage = (
        df.isnull().mean() * 100
    ).sort_values(ascending=False)

    print(missing_percentage.head(10).round(2))

    # Fraud vs legitimate amounts
    print("\n[5] Average Transaction Amount by Fraud Label")

    print(
        df.groupby("isFraud")["TransactionAmt"]
        .mean()
        .round(2)
    )

    print("\n" + "=" * 55)
    print("HERMES DATA EXPLORATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    explore_transactions()
