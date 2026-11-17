
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "validation.csv"

st.set_page_config(
    page_title="Hermes | Transaction Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("Transaction Explorer")
st.caption("Search and inspect historical validation transactions.")


@st.cache_data
def load_transactions():
    return pd.read_csv(DATA_PATH)


df = load_transactions()

transaction_id = st.text_input(
    "Search by Transaction ID",
    placeholder="Enter transaction ID",
)

if transaction_id:
    matching = df[
        df["TransactionID"].astype(str) == transaction_id.strip()
    ]

    if matching.empty:
        st.warning("No matching transaction found.")
    else:
        st.success("Transaction found.")
        st.dataframe(
            matching,
            use_container_width=True,
            hide_index=True,
        )
else:
    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Historical validation data for research and demonstration."
)
