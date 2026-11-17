import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIRECTORY = ROOT / "docs" / "final_evaluation"


st.set_page_config(
    page_title="Hermes | Final Model Evaluation",
    page_icon="🏁",
    layout="wide",
)


@st.cache_data(ttl=60)
def load_final_evaluation_reports(report_directory):
    required_paths = {
        "comparison": report_directory / "model_comparison.csv",
        "confusion": report_directory / "confusion_matrices.csv",
        "metrics": report_directory / "final_test_metrics.json",
        "curves": report_directory / "curve_data.json",
        "methodology": report_directory / "evaluation_methodology.md",
    }
    missing = [path.name for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Final evaluation reports are missing: " + ", ".join(missing))
    return {
        "comparison": pd.read_csv(required_paths["comparison"]),
        "confusion": pd.read_csv(required_paths["confusion"]),
        "metrics": json.loads(required_paths["metrics"].read_text(encoding="utf-8")),
        "curves": json.loads(required_paths["curves"].read_text(encoding="utf-8")),
        "methodology": required_paths["methodology"].read_text(encoding="utf-8"),
    }


st.title("Final Held-Out Model Evaluation")
st.caption("Research/demo-only evaluation on the untouched chronological test split.")
st.warning(
    "This is not live or production monitoring. Scores are uncalibrated model outputs, "
    "not fraud probabilities or confirmed fraud outcomes."
)

try:
    reports = load_final_evaluation_reports(REPORT_DIRECTORY)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python -m ml.evaluation.final_test_evaluation", language="powershell")
    st.stop()

comparison = reports["comparison"]
st.subheader("Fixed-threshold model comparison")
st.dataframe(comparison, use_container_width=True, hide_index=True)

metric_columns = st.columns(len(comparison))
for column, (_, row) in zip(metric_columns, comparison.iterrows(), strict=True):
    column.metric(row["model"], f"PR-AUC {row['pr_auc']:.4f}")
    column.caption(
        f"Threshold {row['decision_threshold']:.6f} | "
        f"Fraud capture {row['fraud_capture_rate']:.2%}"
    )

st.subheader("Confusion matrices")
confusion_columns = st.columns(len(reports["confusion"]))
for column, (_, row) in zip(confusion_columns, reports["confusion"].iterrows(), strict=True):
    matrix = [[row["true_negative_count"], row["false_positive_count"]], [
        row["false_negative_count"], row["true_positive_count"]
    ]]
    figure = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted legitimate", "Predicted alert"],
        y=["Actual legitimate", "Actual fraud"],
        title=row["model"],
        color_continuous_scale="Blues",
    )
    column.plotly_chart(figure, use_container_width=True)

roc_figure = go.Figure()
pr_figure = go.Figure()
for model_name, curves in reports["curves"].items():
    roc = curves["roc"]
    pr = curves["pr"]
    if roc["false_positive_rate"]:
        roc_figure.add_trace(go.Scatter(
            x=roc["false_positive_rate"],
            y=roc["true_positive_rate"],
            mode="lines",
            name=model_name,
        ))
    if pr["recall"]:
        pr_figure.add_trace(go.Scatter(
            x=pr["recall"],
            y=pr["precision"],
            mode="lines",
            name=model_name,
        ))
roc_figure.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate")
pr_figure.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision")
curve_left, curve_right = st.columns(2)
curve_left.plotly_chart(roc_figure, use_container_width=True)
curve_right.plotly_chart(pr_figure, use_container_width=True)

st.subheader("Operating tradeoffs")
st.write(
    "At fixed validation-selected thresholds, false positives represent legitimate transactions "
    "sent for review, while false negatives represent observed fraud missed by the alert rule. "
    "Compare these alongside precision, fraud capture, alert volume, and the review capacity of "
    "this research demonstration. This page does not select a model using test outcomes."
)

with st.expander("Evaluation methodology"):
    st.markdown(reports["methodology"])
