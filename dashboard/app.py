
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = ROOT / "models" / "hermes_xgboost.joblib"

THRESHOLD = 0.7593

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
    page_title="Hermes Fraud Monitoring",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def generate_predictions(_model, data):
    result = data.copy()

    result["FraudScore"] = _model.predict_proba(
        result[FEATURES]
    )[:, 1]

    result["Flagged"] = (
        result["FraudScore"] >= THRESHOLD
    )

    return result


st.title("🛡️ HERMES")
st.caption("Fraud Detection & Risk Monitoring Platform")

st.divider()

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Dataset or trained model not found.")
    st.stop()

model = load_model()
data = load_data()
df = generate_predictions(model, data)

total_transactions = len(df)
actual_fraud = int(df["isFraud"].sum())
flagged_transactions = int(df["Flagged"].sum())
fraud_rate = actual_fraud / total_transactions * 100

st.subheader("Fraud Monitoring Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}",
)

col2.metric(
    "Actual Fraud Cases",
    f"{actual_fraud:,}",
)

col3.metric(
    "Fraud Rate",
    f"{fraud_rate:.2f}%",
)

col4.metric(
    "Flagged Transactions",
    f"{flagged_transactions:,}",
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Actual Fraud Distribution")

    fraud_counts = (
        df["isFraud"]
        .map({0: "Legitimate", 1: "Fraud"})
        .value_counts()
        .reset_index()
    )

    fraud_counts.columns = ["Category", "Count"]

    fig = px.pie(
        fraud_counts,
        names="Category",
        values="Count",
        hole=0.55,
        color="Category",
        color_discrete_map={
            "Legitimate": "#16a34a",
            "Fraud": "#ef4444",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Model Score Distribution")

    fig = px.histogram(
        df,
        x="FraudScore",
        nbins=40,
        color="isFraud",
        labels={
            "FraudScore": "Model Fraud Score",
            "isFraud": "Actual Fraud Label",
        },
        barmode="overlay",
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Suspicious Transaction Investigation")

show_only_flagged = st.checkbox(
    "Show only flagged transactions",
    value=True,
)

filtered = df[df["Flagged"]] if show_only_flagged else df

display_columns = [
    "TransactionID",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "FraudScore",
    "Flagged",
    "isFraud",
]

st.dataframe(
    filtered[display_columns].sort_values(
        "FraudScore",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Research demonstration using historical labeled "
    "validation data. Fraud scores are not calibrated "
    "probabilities. No live banking transactions are used."
)
