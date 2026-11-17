# HERMES Interview Explanation Guide

## Why is fraud detection difficult?

Fraud is rare, patterns change over time, and false positives create real review costs. Accuracy
alone is not enough; a useful operating point balances precision, fraud capture, alert volume, and
the capacity of the review process.

## Why use chronological splits?

Random splits can let later patterns influence earlier training examples. HERMES sorts records by
transaction time, trains on the past, uses a later validation interval for threshold work, and
reserves a later held-out test interval for final read-only evaluation. This better matches a
time-evolving risk problem.

## Why XGBoost?

Gradient-boosted trees capture nonlinear interactions across numerical and encoded categorical
features and provide a strong comparison to a Logistic Regression baseline. The project keeps both
artifacts and evaluates them at fixed thresholds rather than replacing the baseline by assumption.

## How is class imbalance handled?

The baseline uses balanced class weights; XGBoost uses a class-imbalance weight from training data.
Evaluation focuses on PR-AUC, precision, recall/fraud capture, false positives, false negatives,
and alert rate rather than accuracy alone.

## Precision versus recall?

Precision describes how many alerts are observed fraud, while recall/fraud capture describes how
much observed fraud is alerted. Raising a threshold usually reduces review volume and false
positives but can miss more fraud. Thresholds are selected on validation data, then frozen for
test evaluation.

## How do SHAP explanations work here?

The API transforms a transaction with the saved fitted preprocessing pipeline, then uses SHAP's
TreeExplainer on the XGBoost classifier. Contributions are returned on the raw-margin/log-odds
scale and checked against the model raw output. They explain model behavior, not causal evidence.

## Why this API design?

FastAPI centralizes validation, fitted-pipeline inference, explainability, and bounded batch
scoring. Streamlit calls the API instead of loading the serving model for those flows. This keeps
one inference contract and supports automated API tests.

## How does the analyst workflow work?

Creating an investigation persists a scored synthetic transaction intentionally. Analysts can
change only status and notes; updates are transactional and append audit history. Batch scoring is
read-only and never creates investigations automatically.

## What is offline drift monitoring?

The monitoring command compares train-reference feature distributions with validation data using
PSI for numerical fields, total variation distance for categorical fields, and missing-rate
changes. Thresholds are heuristic indicators and explicitly not statistically validated production
alerts.

## How does batch scoring remain safe?

CSV uploads are size- and row-limited, validated row by row, scored in one vectorized model call,
and returned in original order. Invalid rows are retained with errors. Exported CSV values are
escaped against spreadsheet formula injection.

## What do the tests and limitations show?

The suite verifies API contracts, investigation history, SHAP reconstruction, monitoring metrics,
batch validation, fixed-threshold evaluation, and no model-artifact mutation during final
evaluation. Remaining limitations include local-only operation, no authentication or authorization,
uncalibrated scores, historical data, and no production feedback loop.
