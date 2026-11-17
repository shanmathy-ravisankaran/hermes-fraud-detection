
import os

import httpx
import streamlit as st

API_URL = os.getenv(
    "HERMES_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="Hermes | Live Prediction",
    page_icon="🛡️",
    layout="wide",
)

st.title("Live Fraud Prediction")
st.caption(
    "Submit a demonstration transaction to the Hermes "
    "FastAPI fraud detection service."
)

st.info(
    "This is a research demonstration, not a live banking "
    "fraud prevention service. Model scores are not "
    "calibrated fraud probabilities."
)


# Check API availability
try:
    with httpx.Client(timeout=5.0) as client:
        health_response = client.get(f"{API_URL}/health")
        health_response.raise_for_status()

    st.success("Hermes Prediction API is connected.")

except httpx.HTTPError:
    st.error(
        "Cannot connect to the Hermes API. "
        "Start FastAPI on port 8000 before submitting."
    )
    st.stop()


st.subheader("Transaction Information")

with st.form("transaction_form"):

    amount = st.number_input(
        "Transaction Amount (dataset units)",
        min_value=0.01,
        value=5800.0,
        step=10.0,
    )

    transaction_dt = st.number_input(
        "Transaction Time Offset (seconds)",
        min_value=0,
        value=10400000,
        step=3600,
        help=(
            "IEEE-CIS uses a relative time offset, "
            "not a calendar timestamp."
        ),
    )

    product_cd = st.selectbox(
        "Product Category",
        ["W", "C", "R", "H", "S"],
    )

    card4 = st.selectbox(
        "Card Network",
        ["visa", "mastercard", "american express",
         "discover", "missing"],
    )

    card6 = st.selectbox(
        "Card Type",
        ["debit", "credit", "missing"],
    )

    email_domain = st.text_input(
        "Purchaser Email Domain",
        value="gmail.com",
    )

    explain_prediction = st.checkbox(
        "Explain prediction with SHAP",
        help="Requests a local model explanation from FastAPI after scoring.",
    )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    )


if submitted:
    payload = {
        "transaction_amount": amount,
        "transaction_dt": transaction_dt,
        "product_cd": product_cd,
        "card4": card4,
        "card6": card6,
        "email_domain": email_domain.strip() or "missing",
    }

    try:
        with st.spinner("Analyzing transaction..."), httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_URL}/predict",
                json=payload,
            )
            response.raise_for_status()

        result = response.json()

        st.divider()
        st.subheader("Fraud Risk Assessment")

        left, right, third = st.columns(3)

        left.metric(
            "Model Score",
            f"{result['fraud_score']:.4f}",
        )

        right.metric(
            "Risk Index",
            f"{result['risk_score']:.2f}/100",
        )

        third.metric(
            "Decision Threshold",
            f"{result['threshold']:.4f}",
        )

        st.progress(
            min(max(float(result["fraud_score"]), 0.0), 1.0)
        )

        if result["is_suspicious"]:
            st.warning(
                "Transaction flagged for manual review."
            )
        else:
            st.success(
                "No alert at the current model threshold."
            )

        st.write(
            "**Recommendation:**",
            result["recommendation"],
        )

        with st.expander("View API Response"):
            st.json(result)

        if explain_prediction:
            try:
                with st.spinner("Generating SHAP explanation..."), httpx.Client(
                    timeout=30.0
                ) as client:
                    explanation_response = client.post(
                        f"{API_URL}/explain",
                        json=payload,
                    )
                    explanation_response.raise_for_status()

                explanation = explanation_response.json()
                st.subheader("Model Explainability")
                st.caption(
                    "SHAP values explain model behavior for this transaction. They are "
                    "raw-margin (log-odds) contributions, not fraud probabilities or causes."
                )
                explanation_table = [
                    {**item, "direction": "Increases model output"}
                    for item in explanation["top_positive_contributions"]
                ] + [
                    {**item, "direction": "Decreases model output"}
                    for item in explanation["top_negative_contributions"]
                ]
                st.dataframe(
                    explanation_table,
                    use_container_width=True,
                    hide_index=True,
                )
            except httpx.HTTPStatusError as exc:
                st.error(
                    f"Explanation API returned status {exc.response.status_code}: "
                    f"{exc.response.text[:500]}"
                )
            except httpx.RequestError as exc:
                st.error(f"Could not retrieve the explanation: {exc}")

    except httpx.HTTPStatusError as exc:
        st.error(
            f"API returned status "
            f"{exc.response.status_code}: "
            f"{exc.response.text[:500]}"
        )

    except httpx.RequestError as exc:
        st.error(f"Could not contact the API: {exc}")
