import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIRECTORY = ROOT / "docs" / "monitoring"


st.set_page_config(
    page_title="Hermes | Model Monitoring",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=60)
def load_reports(report_directory):
    required_paths = {
        "performance": report_directory / "model_performance_summary.json",
        "drift": report_directory / "feature_drift_summary.csv",
        "amount": report_directory / "transaction_amount_distribution.csv",
        "windows": report_directory / "time_window_performance.csv",
        "methodology": report_directory / "monitoring_methodology.json",
    }
    missing_paths = [path.name for path in required_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Monitoring reports are missing: " + ", ".join(missing_paths)
        )

    with required_paths["performance"].open("r", encoding="utf-8") as report_file:
        performance = json.load(report_file)
    with required_paths["methodology"].open("r", encoding="utf-8") as report_file:
        methodology = json.load(report_file)
    return {
        "performance": performance,
        "drift": pd.read_csv(required_paths["drift"]),
        "amount": pd.read_csv(required_paths["amount"]),
        "windows": pd.read_csv(required_paths["windows"]),
        "methodology": methodology,
    }


st.title("Model Performance & Data Drift Monitoring")
st.caption("Historical offline monitoring using the chronological validation split.")
st.info(
    "This page is not live banking monitoring and does not measure production performance. "
    "Model scores remain uncalibrated; drift severities use heuristic thresholds."
)

try:
    reports = load_reports(REPORT_DIRECTORY)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python -m ml.monitoring.run_monitoring", language="powershell")
    st.stop()

performance = reports["performance"]
drift = reports["drift"]
windows = reports["windows"]

st.subheader("Historical validation performance")
kpi_columns = st.columns(5)
kpi_columns[0].metric("PR-AUC", f"{performance['pr_auc']:.4f}" if performance["pr_auc"] else "n/a")
kpi_columns[1].metric("Fraud capture rate", f"{performance['fraud_capture_rate']:.2%}")
kpi_columns[2].metric("False positives", f"{performance['false_positive_count']:,}")
kpi_columns[3].metric("Alert rate", f"{performance['alert_rate']:.2%}")
kpi_columns[4].metric("F1-score", f"{performance['f1_score']:.4f}")
st.caption(
    f"Dataset: {performance['dataset']} | Period: {performance['evaluation_period']} | "
    f"Fixed decision threshold: {performance['decision_threshold']:.4f}"
)

st.divider()
st.subheader("Feature drift summary")
st.dataframe(
    drift.sort_values(["severity", "feature"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
        "distribution_distance": st.column_config.NumberColumn(
            "Categorical TVD", format="%.4f"
        ),
        "missing_rate_delta": st.column_config.NumberColumn("Missing-rate change", format="%.2%"),
    },
)

severity_chart = drift["severity"].value_counts().rename_axis("severity").reset_index(name="count")
st.plotly_chart(
    px.bar(
        severity_chart,
        x="severity",
        y="count",
        color="severity",
        title="Feature drift severity counts",
        color_discrete_map={"low": "#16a34a", "moderate": "#f59e0b", "high": "#ef4444"},
    ),
    use_container_width=True,
)

st.divider()
st.subheader("Chronological validation windows")
if not windows.empty:
    metric_chart = windows.melt(
        id_vars=["window"],
        value_vars=["precision", "recall", "alert_rate"],
        var_name="metric",
        value_name="value",
    )
    st.plotly_chart(
        px.line(
            metric_chart,
            x="window",
            y="value",
            color="metric",
            markers=True,
            title="Performance and alert rate by chronological window",
        ),
        use_container_width=True,
    )
    st.dataframe(windows, use_container_width=True, hide_index=True)

st.subheader("Transaction amount distribution comparison")
amount_distribution = reports["amount"].copy()
amount_distribution["bin"] = amount_distribution.apply(
    lambda row: f"{row['bin_start']:.0f}–{row['bin_end']:.0f}",
    axis=1,
)
amount_chart = amount_distribution.melt(
    id_vars=["bin"],
    value_vars=["reference_count", "observed_count"],
    var_name="dataset",
    value_name="transaction_count",
)
st.plotly_chart(
    px.bar(
        amount_chart,
        x="bin",
        y="transaction_count",
        color="dataset",
        barmode="group",
        title="Train reference vs validation observation",
    ),
    use_container_width=True,
)

with st.expander("Monitoring methodology and limitations"):
    st.json(reports["methodology"])
