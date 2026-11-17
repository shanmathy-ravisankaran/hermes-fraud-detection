# HERMES Architecture

HERMES is a local research/demo system. It separates model scoring from analyst workflow state and
keeps offline evaluation/reporting outside the serving path.

```mermaid
flowchart LR
    Raw[Historical transaction data] --> Validate[Data validation and feature engineering]
    Validate --> Split[Chronological train / validation / held-out test splits]
    Split --> Train[Pipeline training\nLogistic Regression and XGBoost]
    Train --> Artifacts[Saved sklearn model artifacts]

    Artifacts --> API[FastAPI local services]
    API --> Predict[Single and batch inference]
    API --> Explain[SHAP explanation service]
    API --> Investigations[SQLite investigation workflow\nstatus, notes, audit history]

    API --> Dashboard[Streamlit dashboard]
    Dashboard --> Explorer[Explorer, Analytics, Live Prediction]
    Dashboard --> Analyst[Investigation Center and Batch Scoring]
    Dashboard --> Reports[Monitoring and Final Evaluation pages]

    Split --> Monitor[Offline monitoring\ntrain reference vs validation]
    Split --> Final[Final read-only evaluation\nheld-out test only]
    Monitor --> Reports
    Final --> Reports
```

## Boundaries

- **Model artifacts:** saved sklearn pipelines contain fitted preprocessing and classifiers. Serving
  and batch scoring call those pipelines directly; they do not retrain or refit them.
- **FastAPI:** local interface for scoring, explainability, batch uploads, and analyst workflows.
- **SQLite:** stores only local investigation workflow state. Batch scoring remains non-persistent.
- **Streamlit:** calls FastAPI for prediction, investigation, and explanation features; it does not
  load the serving model for those actions.
- **Offline jobs:** monitoring uses train/validation data; final evaluation uses held-out test data
  with fixed validation-selected thresholds. Reports contain aggregate metrics only.
