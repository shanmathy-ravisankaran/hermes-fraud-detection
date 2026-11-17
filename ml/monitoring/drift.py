"""Feature distribution monitoring for historical offline datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERICAL_PSI_THRESHOLDS = {"moderate": 0.10, "high": 0.25}
CATEGORICAL_TVD_THRESHOLDS = {"moderate": 0.10, "high": 0.20}
MISSING_RATE_THRESHOLDS = {"moderate": 0.02, "high": 0.05}

SEVERITY_RANK = {"low": 0, "moderate": 1, "high": 2}


def severity_from_value(value: float, thresholds: dict[str, float]) -> str:
    if value >= thresholds["high"]:
        return "high"
    if value >= thresholds["moderate"]:
        return "moderate"
    return "low"


def combine_severity(*severities: str) -> str:
    return max(severities, key=lambda item: SEVERITY_RANK[item])


def population_stability_index(
    reference: pd.Series,
    observed: pd.Series,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI with quantile reference bins and safe zero-frequency handling."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    observed_values = pd.to_numeric(observed, errors="coerce").dropna().to_numpy()

    if len(reference_values) == 0 or len(observed_values) == 0:
        return 0.0

    bin_edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(bin_edges) < 2:
        reference_constant = float(reference_values[0])
        observed_constant = np.allclose(observed_values, reference_constant)
        return 0.0 if observed_constant else 1.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    observed_counts, _ = np.histogram(observed_values, bins=bin_edges)
    reference_rates = np.clip(reference_counts / len(reference_values), epsilon, None)
    observed_rates = np.clip(observed_counts / len(observed_values), epsilon, None)

    return float(np.sum((observed_rates - reference_rates) * np.log(observed_rates / reference_rates)))


def categorical_distribution_drift(
    reference: pd.Series,
    observed: pd.Series,
) -> tuple[float, float]:
    """Return total variation distance and observed unseen-category rate."""
    reference_values = reference.fillna("__missing__").astype(str)
    observed_values = observed.fillna("__missing__").astype(str)
    if reference_values.empty or observed_values.empty:
        return 0.0, 0.0

    reference_rates = reference_values.value_counts(normalize=True)
    observed_rates = observed_values.value_counts(normalize=True)
    categories = reference_rates.index.union(observed_rates.index)
    distance = 0.5 * sum(
        abs(reference_rates.get(category, 0.0) - observed_rates.get(category, 0.0))
        for category in categories
    )
    unseen_rate = float((~observed_values.isin(reference_rates.index)).mean())
    return float(distance), unseen_rate


def transaction_amount_distribution(
    reference: pd.Series,
    observed: pd.Series,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Produce aggregate amount-bin counts without reporting raw transactions."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    observed_values = pd.to_numeric(observed, errors="coerce").dropna().to_numpy()
    if len(reference_values) == 0 or len(observed_values) == 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "reference_count", "observed_count"])

    bin_edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(bin_edges) < 2:
        bin_edges = np.array([reference_values[0] - 0.5, reference_values[0] + 0.5])
    else:
        bin_edges[0] = min(bin_edges[0], observed_values.min())
        bin_edges[-1] = max(bin_edges[-1], observed_values.max())

    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    observed_counts, _ = np.histogram(observed_values, bins=bin_edges)
    return pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "reference_count": reference_counts,
        "observed_count": observed_counts,
    })


def build_feature_drift_summary(
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    numerical_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Build per-feature drift summaries for fitted model inputs only."""
    rows = []
    for feature in numerical_features + categorical_features:
        if feature not in reference.columns or feature not in observed.columns:
            continue

        reference_missing_rate = float(reference[feature].isna().mean())
        observed_missing_rate = float(observed[feature].isna().mean())
        missing_rate_delta = abs(observed_missing_rate - reference_missing_rate)
        missing_severity = severity_from_value(missing_rate_delta, MISSING_RATE_THRESHOLDS)

        row = {
            "feature": feature,
            "reference_missing_rate": reference_missing_rate,
            "observed_missing_rate": observed_missing_rate,
            "missing_rate_delta": missing_rate_delta,
            "missing_severity": missing_severity,
        }
        if feature in numerical_features:
            psi = population_stability_index(reference[feature], observed[feature])
            distribution_severity = severity_from_value(psi, NUMERICAL_PSI_THRESHOLDS)
            row.update({
                "feature_type": "numerical",
                "psi": psi,
                "distribution_distance": np.nan,
                "unseen_category_rate": np.nan,
                "distribution_severity": distribution_severity,
                "severity": combine_severity(distribution_severity, missing_severity),
            })
        else:
            distance, unseen_rate = categorical_distribution_drift(
                reference[feature],
                observed[feature],
            )
            distribution_severity = severity_from_value(distance, CATEGORICAL_TVD_THRESHOLDS)
            row.update({
                "feature_type": "categorical",
                "psi": np.nan,
                "distribution_distance": distance,
                "unseen_category_rate": unseen_rate,
                "distribution_severity": distribution_severity,
                "severity": combine_severity(distribution_severity, missing_severity),
            })
        rows.append(row)

    return pd.DataFrame(rows)
