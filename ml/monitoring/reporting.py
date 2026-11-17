"""Non-destructive aggregate report generation for offline monitoring."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def write_monitoring_reports(
    output_dir: Path,
    *,
    feature_drift: pd.DataFrame,
    amount_distribution: pd.DataFrame,
    performance: dict[str, Any],
    time_windows: pd.DataFrame,
    methodology: dict[str, Any],
) -> dict[str, Path]:
    """Write aggregate CSV/JSON reports without transaction-level rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_drift": output_dir / "feature_drift_summary.csv",
        "missing_value_drift": output_dir / "missing_value_drift.csv",
        "amount_distribution": output_dir / "transaction_amount_distribution.csv",
        "model_performance": output_dir / "model_performance_summary.json",
        "time_window_performance": output_dir / "time_window_performance.csv",
        "methodology": output_dir / "monitoring_methodology.json",
    }
    feature_drift.to_csv(paths["feature_drift"], index=False)
    feature_drift.loc[:, [
        "feature",
        "reference_missing_rate",
        "observed_missing_rate",
        "missing_rate_delta",
        "missing_severity",
    ]].to_csv(paths["missing_value_drift"], index=False)
    amount_distribution.to_csv(paths["amount_distribution"], index=False)
    time_windows.to_csv(paths["time_window_performance"], index=False)

    with paths["model_performance"].open("w", encoding="utf-8") as report_file:
        json.dump(performance, report_file, indent=2, sort_keys=True)

    methodology_report = {
        **methodology,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "report_scope": "historical offline monitoring only",
    }
    with paths["methodology"].open("w", encoding="utf-8") as report_file:
        json.dump(methodology_report, report_file, indent=2, sort_keys=True)

    return paths
