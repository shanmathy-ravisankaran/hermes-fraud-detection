# HERMES Fraud Detection & Risk Monitoring Platform

HERMES is a local, research-oriented fraud-risk demonstration built around historical IEEE-CIS
transaction data. It shows an end-to-end workflow: chronological data preparation, model
training, API inference, analyst investigation tracking, SHAP explainability, offline monitoring,
batch scoring, and final held-out evaluation.

> **Research/demo only.** This project is not production-ready, does not connect to banks or
> payment networks, and does not make fraud determinations. Model scores are uncalibrated outputs,
> not fraud probabilities or confirmed fraud outcomes.

## Business problem

Fraud review teams need to prioritize transactions with limited analyst capacity. HERMES explores
that tradeoff by producing a model alert at a fixed threshold, retaining analyst workflow state
separately from the model result, and reporting false-positive and missed-fraud outcomes on
historical labeled data.

## Architecture and stack

- Python 3.11, pandas, NumPy, scikit-learn, and XGBoost
- FastAPI for local scoring, explainability, investigation, and batch APIs
- Streamlit and Plotly for the local demonstration dashboard
- SQLite for local investigation status, notes, and audit history
- SHAP TreeExplainer for local XGBoost explanations
- pytest and Ruff for automated verification and linting

See [architecture documentation](docs/architecture.md) for the component diagram.

## Data pipeline and split strategy

The project engineers these model inputs from historical transaction data:

`TransactionAmt`, `TransactionAmtLog`, `TransactionHour`, `TransactionDay`, `ProductCD`, `card4`,
`card6`, and `P_emaildomain`.

The source data is ordered chronologically before being split into train (70%), validation (15%),
and held-out test (15%) datasets. Training fits preprocessing within each sklearn pipeline. The
validation split supports threshold selection and offline monitoring; `test.csv` is reserved for
the final, read-only evaluation.

Raw and processed datasets are intentionally ignored by Git. Supply the required local artifacts
before running the full workflow.

## Generating local artifacts

Model artifacts and datasets are deliberately not committed. With the original local raw dataset
available at `data/raw/train_transaction.csv`, the project provides these existing module entry
points to regenerate ignored local artifacts:

```powershell
& .\.venv\Scripts\python.exe -m ml.features.build_features
& .\.venv\Scripts\python.exe -m ml.training.train_baseline
& .\.venv\Scripts\python.exe -m ml.evaluation.optimize_threshold
& .\.venv\Scripts\python.exe -m ml.training.train_xgboost
```

These commands train or overwrite ignored local model/report artifacts and are not part of normal
API/dashboard startup. The final evaluator expects those locally generated artifacts and does not
train models itself.

## Models and fixed operating thresholds

Two saved sklearn pipelines are evaluated:

- Logistic Regression baseline — fixed validation-selected threshold: `0.85`
- XGBoost — fixed validation-selected threshold: `0.7593424320220947`

Neither model is retrained, calibrated, tuned, or selected with held-out test labels. Their fitted
preprocessing pipelines are used directly for both single and batch inference.

## Final held-out test results

The final evaluator uses the untouched chronological `data/processed/test.csv` split with the
fixed thresholds above.

| Model | PR-AUC | F1 | Precision | Fraud capture | Alerts |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1356 | 0.1888 | 0.1428 | 27.86% | 6,017 |
| XGBoost | 0.1586 | 0.2410 | 0.1927 | 32.18% | 5,148 |

These values document one historical held-out evaluation, not expected production performance.
Aggregate reports are available under [docs/final_evaluation](docs/final_evaluation).

## Application capabilities

The Streamlit app includes:

- **Transaction Explorer** — searches historical validation transactions.
- **Fraud Analytics** — explores validation metrics and score distributions.
- **Live Prediction** — scores one demonstration transaction through FastAPI.
- **Investigation Center** — manages local analyst status, notes, history, and SHAP explanations.
- **Model Monitoring** — displays historical offline drift and validation-performance reports.
- **Batch Scoring** — validates and scores bounded CSV uploads without persisting investigations.
- **Final Model Evaluation** — presents aggregate held-out test comparison reports.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Local API health check. |
| `POST /predict` | Score one transaction. |
| `POST /explain` | Return a local SHAP explanation for one transaction. |
| `POST /predict/batch` | Validate and score a bounded CSV upload without saving records. |
| `POST /investigations` | Score and save an intentionally created investigation. |
| `GET /investigations` | List recent investigations with optional status filtering. |
| `GET/PATCH /investigations/{id}` | Retrieve or update analyst status and notes. |
| `GET /investigations/{id}/history` | Retrieve audit history. |
| `GET /investigations/{id}/explanation` | Explain the stored transaction without mutation. |

Open local Swagger UI at `http://127.0.0.1:8000/docs` after starting FastAPI.

## Local setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Start the local API:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

In another PowerShell window, start Streamlit:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Optional local API override for dashboard pages:

```powershell
$env:HERMES_API_URL = "http://127.0.0.1:8000"
```

## Offline monitoring and final evaluation

Generate historical validation monitoring reports:

```powershell
& .\.venv\Scripts\python.exe -m ml.monitoring.run_monitoring
```

Run the final held-out evaluation:

```powershell
& .\.venv\Scripts\python.exe -m ml.evaluation.final_test_evaluation
```

The final evaluator writes aggregate-only reports to `docs/final_evaluation/`:

- `model_comparison.csv`
- `confusion_matrices.csv`
- `final_test_metrics.json`
- `curve_data.json`
- `evaluation_methodology.md`

## Testing

```powershell
& .\.venv\Scripts\python.exe -m pytest -v
& .\.venv\Scripts\python.exe -m ruff check backend ml dashboard tests
```

## Limitations and future improvements

- Historical offline results are not production monitoring or real-world fraud impact estimates.
- Fraud labels are highly imbalanced, and threshold tradeoffs require operational review capacity.
- No authentication, authorization, deployment hardening, rate limiting, or production observability
  is implemented; keep the API local-only.
- SHAP contributions describe model behavior, not fraud causes.
- Future work could add authenticated users, protected model/data registries, calibrated scores,
  robust monitoring alerts, human-review feedback loops, and governed deployment workflows.

For presentation material, see the [portfolio overview](docs/portfolio_overview.md),
[demo walkthrough](docs/demo_walkthrough.md), and [interview guide](docs/interview_explanation.md).
