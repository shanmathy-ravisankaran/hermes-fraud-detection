# HERMES System Architecture

This diagram reflects the implemented local research/demo workflow. It intentionally separates
historical/offline jobs from local FastAPI serving and SQLite-backed analyst workflow state.

```mermaid
flowchart LR
    Raw[Historical transaction data] --> Features[Validation and feature engineering]
    Features --> Split[Chronological splits<br/>train / validation / held-out test]

    Split --> Train[Training scripts<br/>Logistic Regression + XGBoost]
    Train --> Artifacts[Ignored local sklearn model artifacts]

    Artifacts --> API[FastAPI local service]
    API --> Single[Single prediction]
    API --> Batch[Bounded CSV batch scoring]
    API --> Shap[SHAP explanation service]
    API --> SQLite[(Local SQLite investigations<br/>status, notes, audit history)]

    API --> Streamlit[Streamlit dashboard]
    Streamlit --> Analyst[Live prediction + Investigation Center]
    Streamlit --> Explore[Explorer + Analytics + Batch Scoring]

    Split --> Monitoring[Offline monitoring<br/>train reference vs validation]
    Split --> Final[Read-only final evaluation<br/>held-out test]
    Monitoring --> Reports[Aggregate reports]
    Final --> Reports
    Reports --> Streamlit
```

## Rendering note

This repository includes Mermaid source so GitHub can render it directly. No Mermaid CLI renderer
is installed locally, so no PNG or SVG export is included. If an SVG is later generated from this
exact source, save it beside this file as `system_architecture.svg`.
