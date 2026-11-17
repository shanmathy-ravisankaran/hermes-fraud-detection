import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.investigation_api import (
    InvestigationApiConnectionError,
    InvestigationApiResponseError,
    score_batch_file,
)
from ml.batch.reporting import batch_results_to_csv

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "data" / "samples" / "batch_scoring_template.csv"
API_URL = os.getenv("HERMES_API_URL", "http://127.0.0.1:8000").rstrip("/")


st.set_page_config(
    page_title="Hermes | Batch Scoring",
    page_icon="📥",
    layout="wide",
)


def show_api_error(error):
    if isinstance(error, InvestigationApiConnectionError):
        st.error("Cannot connect to the local Hermes API. Start FastAPI on port 8000.")
    elif error.status_code == 413:
        st.error("The file exceeds the configured local-development upload limit.")
    elif error.status_code == 422:
        st.error(f"The batch was rejected: {error.detail}")
    else:
        st.error(f"Batch scoring failed: {error.detail}")


st.title("Batch Fraud Scoring")
st.caption("Score demonstration transaction CSV files through the local FastAPI service.")
st.info(
    "Batch scoring is read-only: it does not create investigations. Scores are uncalibrated "
    "model outputs, and flagged transactions are not confirmed fraud."
)

st.subheader("CSV input schema")
st.code(
    "source_row_id,transaction_amount,transaction_dt,product_cd,card4,card6,email_domain",
    language="text",
)
st.caption(
    "Required: transaction_amount, transaction_dt, product_cd. Optional: card4, card6, "
    "email_domain, source_row_id. Local-development limits default to 500 rows and 1 MiB."
)
if TEMPLATE_PATH.is_file():
    st.download_button(
        "Download synthetic sample CSV",
        data=TEMPLATE_PATH.read_bytes(),
        file_name="hermes_batch_scoring_template.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
if st.button("Validate and score batch", type="primary", disabled=uploaded_file is None):
    try:
        with st.spinner("Validating and scoring valid rows in one model request..."):
            st.session_state["batch_scoring_result"] = score_batch_file(
                API_URL,
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
    except (InvestigationApiConnectionError, InvestigationApiResponseError) as exc:
        show_api_error(exc)

batch_result = st.session_state.get("batch_scoring_result")
if not batch_result:
    st.stop()

summary = batch_result["summary"]
st.subheader("Batch summary")
kpi_columns = st.columns(5)
kpi_columns[0].metric("Uploaded rows", f"{summary['total_uploaded_rows']:,}")
kpi_columns[1].metric("Valid rows", f"{summary['valid_rows']:,}")
kpi_columns[2].metric("Invalid rows", f"{summary['invalid_rows']:,}")
kpi_columns[3].metric("Scored rows", f"{summary['successfully_scored_rows']:,}")
kpi_columns[4].metric("Flagged", f"{summary['flagged_transactions']:,}")
st.caption(
    f"Flagged percentage: {summary['flagged_percentage']:.2f}% | "
    f"Average uncalibrated model score: "
    f"{summary['average_model_score']:.4f}" if summary["average_model_score"] is not None else
    "No valid rows were scored."
)
st.caption(f"Processing duration: {summary['processing_duration_ms']:.2f} ms")

if batch_result["unexpected_columns_ignored"]:
    st.warning(
        "Ignored unexpected CSV columns: " + ", ".join(batch_result["unexpected_columns_ignored"])
    )

results = batch_result["results"]
scored_results = [result for result in results if result["validation_status"] == "scored"]
if scored_results:
    score_data = pd.DataFrame(scored_results)
    st.plotly_chart(
        px.histogram(
            score_data,
            x="model_score",
            color="is_flagged",
            nbins=20,
            title="Uncalibrated model score distribution",
            labels={"model_score": "Model score", "is_flagged": "Flagged"},
        ),
        use_container_width=True,
    )

st.subheader("Batch results")
st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
st.download_button(
    "Download scored results CSV",
    data=batch_results_to_csv(results),
    file_name="hermes_batch_results.csv",
    mime="text/csv",
)

validation_errors = batch_result["validation_errors"]
if validation_errors:
    st.subheader("Invalid-row validation errors")
    st.dataframe(pd.DataFrame(validation_errors), use_container_width=True, hide_index=True)
    st.download_button(
        "Download validation errors CSV",
        data=batch_results_to_csv(validation_errors),
        file_name="hermes_batch_validation_errors.csv",
        mime="text/csv",
    )
