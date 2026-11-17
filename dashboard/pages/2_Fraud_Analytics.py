
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = ROOT / "models" / "hermes_xgboost.joblib"

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

st.set_page_config(
    page_title="Hermes | Fraud Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Fraud Detection Analytics")
st.caption(
    "XGBoost model performance on historical validation data"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_validation_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def calculate_scores(_model, data):
    return _model.predict_proba(data[FEATURES])[:, 1]


if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or validation dataset not found.")
    st.stop()

model = load_model()
df = load_validation_data()

scores = calculate_scores(model, df)

st.sidebar.header("Model Settings")

threshold = st.sidebar.slider(
    "Fraud Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.75,
    step=0.01,
)

predictions = (scores >= threshold).astype(int)
actual = df["isFraud"]

precision = precision_score(
    actual, predictions, zero_division=0
)

recall = recall_score(
    actual, predictions, zero_division=0
)

f1 = f1_score(
    actual, predictions, zero_division=0
)

pr_auc = average_precision_score(actual, scores)
roc_auc = roc_auc_score(actual, scores)

tn, fp, fn, tp = confusion_matrix(
    actual, predictions, labels=[0, 1]
).ravel()

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precision", f"{precision:.2%}")
col2.metric("Recall", f"{recall:.2%}")
col3.metric("F1-Score", f"{f1:.4f}")
col4.metric("PR-AUC", f"{pr_auc:.4f}")

st.caption(f"ROC-AUC: {roc_auc:.4f}")

st.divider()

st.subheader("Fraud Classification Outcomes")

outcomes = pd.DataFrame({
    "Outcome": [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ],
    "Count": [
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    ],
})

fig = px.bar(
    outcomes,
    x="Outcome",
    y="Count",
    color="Outcome",
    text="Count",
    title="Confusion Matrix Breakdown",
)

st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    st.metric("Fraud Detected", f"{tp:,}")
    st.metric("Fraud Missed", f"{fn:,}")

with right:
    st.metric("False Alarms", f"{fp:,}")
    st.metric("Correctly Legitimate", f"{tn:,}")

st.divider()

st.subheader("Fraud Score Distribution")

score_df = pd.DataFrame({
    "Fraud Score": scores,
    "Actual Label": actual.map({
        0: "Legitimate",
        1: "Fraud",
    }),
})

fig = px.histogram(
    score_df,
    x="Fraud Score",
    color="Actual Label",
    nbins=50,
    barmode="overlay",
    opacity=0.7,
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Threshold changes are exploratory and use validation "
    "data only. The deployed API currently uses threshold "
    "0.7593. Model scores are not calibrated probabilities."
)
