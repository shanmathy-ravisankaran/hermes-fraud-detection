# Hermes Final Held-Out Test Evaluation

Evaluation date: 2026-11-17T04:11:59.953344+00:00

## Scope

This report evaluates the immutable chronological `data/processed/test.csv` split only. No
model was retrained, refit, calibrated, tuned, or selected with test labels. The Logistic
Regression baseline uses its validation-selected threshold of 0.85. XGBoost
uses the full persisted validation threshold of 0.7593424320220947, loaded from
`docs/xgboost_validation_metrics.csv`.

## Metrics

Precision is TP/(TP+FP), recall/fraud capture is TP/(TP+FN), F1 is the harmonic mean of
precision and recall, and alert rate is alerts divided by all test transactions. ROC-AUC and
PR-AUC use score ranking only; operating metrics use the fixed thresholds above.

## Limitations

This is a research/demo evaluation, not production performance. Scores are uncalibrated model
outputs and are not fraud probabilities or confirmed fraud outcomes. Curves contain bounded,
aggregate points only; no transaction-level test data is exported.
