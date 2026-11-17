# HERMES Portfolio Overview

## Problem

HERMES demonstrates a fraud-risk workflow for prioritizing historical transactions for review.
The objective is not to declare fraud automatically; it is to show the technical tradeoffs between
fraud capture, false positives, analyst workflow, and model transparency.

## Architecture and features

The project combines chronological data preparation, sklearn preprocessing pipelines, Logistic
Regression and XGBoost artifacts, FastAPI, SQLite investigation tracking, Streamlit, SHAP,
historical monitoring, and bounded batch scoring. See [architecture.md](architecture.md).

## Evaluation

The final held-out chronological test evaluation uses thresholds fixed on validation data:

- Logistic Regression: threshold 0.85, PR-AUC 0.1356, F1 0.1888, fraud capture 27.86%.
- XGBoost: threshold 0.7593424320220947, PR-AUC 0.1586, F1 0.2410, fraud capture 32.18%.

These are descriptive held-out test results, not a claim of production readiness or business
impact. Full aggregate metrics are in [final_evaluation](final_evaluation).

## Strengths demonstrated

- Time-aware split discipline and fixed-threshold evaluation.
- A fitted preprocessing pipeline reused consistently by API, SHAP, and batch scoring.
- Analyst status, notes, and atomic audit history separate from model predictions.
- Local explainability, offline drift reporting, and safe bounded batch ingestion.
- Automated tests across API, investigations, explainability, monitoring, batch scoring, and final
  evaluation.

## Limitations

This remains a local research demo. It has no production authentication, authorization, secure
deployment, calibrated probabilities, real-time feature pipeline, or live fraud feedback loop.
