import os

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.investigation_api import (
    InvestigationApiConnectionError,
    InvestigationApiResponseError,
    get_investigation,
    get_investigation_explanation,
    get_investigation_history,
    get_investigations,
    update_investigation,
)

API_URL = os.getenv("HERMES_API_URL", "http://127.0.0.1:8000").rstrip("/")
STATUSES = ["pending", "under_review", "confirmed_fraud", "false_positive"]
KPI_FETCH_LIMIT = 500


st.set_page_config(
    page_title="Hermes | Investigation Center",
    page_icon="🗂️",
    layout="wide",
)


@st.cache_data(ttl=10, show_spinner=False)
def load_investigations(api_url, limit, investigation_status):
    return get_investigations(
        api_url,
        limit=limit,
        investigation_status=investigation_status,
    )


@st.cache_data(ttl=10, show_spinner=False)
def load_investigation(api_url, investigation_id):
    return get_investigation(api_url, investigation_id)


@st.cache_data(ttl=10, show_spinner=False)
def load_history(api_url, investigation_id):
    return get_investigation_history(api_url, investigation_id)


@st.cache_data(ttl=10, show_spinner=False)
def load_explanation(api_url, investigation_id):
    return get_investigation_explanation(api_url, investigation_id)


def show_api_error(error):
    if isinstance(error, InvestigationApiConnectionError):
        st.error(
            "Cannot connect to the Hermes API. Start FastAPI locally on port 8000 "
            "or set HERMES_API_URL."
        )
    elif error.status_code == 404:
        st.error("The selected investigation no longer exists. Refresh the investigation list.")
    elif error.status_code == 422:
        st.error(f"The API rejected the request: {error.detail}")
    else:
        st.error(f"API request failed: {error.detail}")


def status_count(investigations, investigation_status):
    return sum(item.get("status") == investigation_status for item in investigations)


st.title("Investigation Center")
st.caption("Local demonstration workflow for analyst review and audit history.")
st.info(
    "Model scores are uncalibrated model scores, not fraud probabilities. "
    "Keep this local-only analyst interface behind authentication and authorization "
    "before any non-demonstration deployment."
)

if "investigation_update_message" in st.session_state:
    st.success(st.session_state.pop("investigation_update_message"))

try:
    kpi_response = load_investigations(API_URL, KPI_FETCH_LIMIT, None)
except (InvestigationApiConnectionError, InvestigationApiResponseError) as exc:
    show_api_error(exc)
    st.stop()

kpi_records = kpi_response.get("investigations", [])
kpi_columns = st.columns(5)
kpi_columns[0].metric("Investigations fetched", f"{len(kpi_records):,}")
kpi_columns[1].metric("Pending", f"{status_count(kpi_records, 'pending'):,}")
kpi_columns[2].metric("Under review", f"{status_count(kpi_records, 'under_review'):,}")
kpi_columns[3].metric("Confirmed fraud", f"{status_count(kpi_records, 'confirmed_fraud'):,}")
kpi_columns[4].metric("False positive", f"{status_count(kpi_records, 'false_positive'):,}")
st.caption(
    f"KPI counts reflect the latest {len(kpi_records):,} records fetched from the API "
    f"(maximum {KPI_FETCH_LIMIT}). They are not a global total when more records exist."
)

st.divider()
st.subheader("Search and filter investigations")

filter_left, filter_center, filter_right, filter_limit = st.columns(4)
search_id = filter_left.text_input("Investigation ID", placeholder="Search fetched records")
selected_status = filter_center.selectbox("Status", ["all", *STATUSES])
suspicious_only = filter_right.checkbox("Suspicious transactions only")
result_limit = filter_limit.selectbox("Result limit", [25, 50, 100, 250, 500], index=2)
minimum_score = st.slider("Minimum model score", 0.0, 1.0, 0.0, 0.01)

try:
    list_response = load_investigations(
        API_URL,
        result_limit,
        None if selected_status == "all" else selected_status,
    )
except (InvestigationApiConnectionError, InvestigationApiResponseError) as exc:
    show_api_error(exc)
    st.stop()

fetched_records = list_response.get("investigations", [])
filtered_records = [
    record
    for record in fetched_records
    if (not search_id.strip() or str(record.get("id")) == search_id.strip())
    and (not suspicious_only or bool(record.get("is_suspicious")))
    and float(record.get("fraud_score", 0)) >= minimum_score
]

st.caption(
    f"Showing {len(filtered_records):,} locally filtered record(s) from "
    f"{len(fetched_records):,} record(s) fetched from the API. Status and limit are "
    "server-side; ID, suspicious-only, and score filters apply only to fetched records."
)

table_columns = [
    "id",
    "status",
    "transaction_amount",
    "fraud_score",
    "threshold",
    "is_suspicious",
    "model_name",
    "created_at",
]
table_data = pd.DataFrame(filtered_records)
if table_data.empty:
    st.warning("No investigations match the current filters.")
else:
    st.dataframe(
        table_data.reindex(columns=table_columns),
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_score": st.column_config.NumberColumn("Model score", format="%.4f"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.4f"),
            "is_suspicious": st.column_config.CheckboxColumn("Suspicious"),
            "created_at": "Created",
        },
    )

selection_options = [None, *[record["id"] for record in filtered_records]]
selected_id = st.selectbox(
    "Select an investigation for review",
    selection_options,
    format_func=lambda item: "Choose an investigation" if item is None else f"Investigation #{item}",
)

if selected_id is None:
    st.stop()

try:
    investigation = load_investigation(API_URL, selected_id)
    history_response = load_history(API_URL, selected_id)
    explanation_response = load_explanation(API_URL, selected_id)
except (InvestigationApiConnectionError, InvestigationApiResponseError) as exc:
    show_api_error(exc)
    st.stop()

st.divider()
st.subheader(f"Investigation #{investigation['id']}")

details_left, details_center, details_right = st.columns(3)
details_left.metric("Transaction amount", f"{investigation['transaction_amount']:,.2f}")
details_center.metric("Model score", f"{investigation['fraud_score']:.4f}")
details_right.metric("Decision threshold", f"{investigation['threshold']:.4f}")
st.progress(min(max(float(investigation["fraud_score"]), 0.0), 1.0))

if investigation["is_suspicious"]:
    st.warning("Model decision: suspicious — manual review recommended.")
else:
    st.success("Model decision: not suspicious at the configured threshold.")

transaction_details = {
    "product_cd": investigation.get("product_cd"),
    "card4": investigation.get("card4"),
    "card6": investigation.get("card6"),
    "email_domain": investigation.get("email_domain"),
    "transaction_dt": investigation.get("transaction_dt"),
    "model_name": investigation.get("model_name"),
    "created_at": investigation.get("created_at"),
}
with st.expander("Transaction and model details", expanded=True):
    st.json(transaction_details)

st.subheader("Model Explainability")
st.caption(
    "These SHAP contributions explain how the model reached this output; they do not "
    "identify confirmed fraud causes. Values are raw-margin (log-odds) contributions, "
    "not percentages or fraud probabilities."
)
explanation_metrics = st.columns(3)
explanation_metrics[0].metric("Model score", f"{explanation_response['model_score']:.4f}")
explanation_metrics[1].metric("Decision threshold", f"{explanation_response['threshold']:.4f}")
explanation_metrics[2].metric("Raw model output", f"{explanation_response['raw_model_output']:.4f}")

positive_contributions = explanation_response.get("top_positive_contributions", [])
negative_contributions = explanation_response.get("top_negative_contributions", [])
contribution_rows = [
    {**item, "direction": "Increases model output"}
    for item in positive_contributions
] + [
    {**item, "direction": "Decreases model output"}
    for item in negative_contributions
]

if contribution_rows:
    contribution_data = pd.DataFrame(contribution_rows)
    contribution_data["display_feature"] = contribution_data.apply(
        lambda row: f"{row['original_feature']} ({row['feature_value']:.0f})",
        axis=1,
    )
    chart = px.bar(
        contribution_data.sort_values("shap_value"),
        x="shap_value",
        y="display_feature",
        color="direction",
        orientation="h",
        labels={"shap_value": "SHAP contribution (raw margin)", "display_feature": "Feature"},
        color_discrete_map={
            "Increases model output": "#ef4444",
            "Decreases model output": "#16a34a",
        },
    )
    st.plotly_chart(chart, use_container_width=True)
    positive_column, negative_column = st.columns(2)
    positive_column.dataframe(
        pd.DataFrame(positive_contributions),
        use_container_width=True,
        hide_index=True,
    )
    negative_column.dataframe(
        pd.DataFrame(negative_contributions),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No non-zero SHAP contributions were returned for this transaction.")

st.subheader("Analyst review")
current_status = investigation["status"]
existing_notes = investigation.get("analyst_notes") or ""
with st.form(f"review_form_{selected_id}"):
    updated_status = st.selectbox(
        "Investigation status",
        STATUSES,
        index=STATUSES.index(current_status),
    )
    updated_notes = st.text_area(
        "Analyst notes",
        value=existing_notes,
        max_chars=2000,
        help="Notes are only sent when their content changes. Clear this field and save to remove notes.",
    )
    save_changes = st.form_submit_button("Save changes", type="primary")

if save_changes:
    notes_changed = updated_notes != existing_notes
    if updated_status == current_status and not notes_changed:
        st.info("No changes to save.")
    else:
        try:
            update_investigation(
                API_URL,
                selected_id,
                updated_status,
                analyst_notes=updated_notes,
                notes_changed=notes_changed,
            )
        except (InvestigationApiConnectionError, InvestigationApiResponseError) as exc:
            show_api_error(exc)
        else:
            load_investigations.clear()
            load_investigation.clear()
            load_history.clear()
            load_explanation.clear()
            st.session_state["investigation_update_message"] = (
                f"Investigation #{selected_id} updated successfully."
            )
            st.rerun()

st.subheader("Audit history")
history = history_response.get("history", [])
if history:
    history_table = pd.DataFrame(history).reindex(
        columns=[
            "created_at",
            "previous_status",
            "new_status",
            "previous_notes",
            "updated_notes",
        ]
    )
    st.dataframe(history_table, use_container_width=True, hide_index=True)
else:
    st.caption("No status or note changes have been recorded yet.")
