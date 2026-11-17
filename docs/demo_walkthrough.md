# Five-Minute Demo Walkthrough

## 0:00–0:30 — Frame the demo

Open the HERMES Streamlit home page. Explain that it is a local research/demo platform using
historical labeled data, not a live banking system. State that scores are uncalibrated and alerts
are review priorities, not fraud decisions.

## 0:30–1:15 — Analytics and data exploration

Open **Fraud Analytics** and point out the validation-data metrics and score distribution. Then
open **Transaction Explorer** to show that historical records can be inspected without presenting
customer identities.

## 1:15–2:00 — Live scoring and SHAP

Open **Live Prediction**, enter a synthetic transaction, and select **Explain prediction with
SHAP**. Explain that the API owns inference and SHAP generation, while Streamlit visualizes the
response. Clarify that contributions describe model behavior, not fraud causes.

## 2:00–3:00 — Analyst workflow

Use Swagger `POST /investigations` to create a synthetic investigation, then open
**Investigation Center**. Update a status and an analyst note deliberately, save, and show the
chronological audit history and explanation chart.

## 3:00–3:45 — Batch scoring

Open **Batch Scoring**, download the synthetic template, upload it, and run validation/scoring.
Highlight row-level errors, bounded upload limits, vectorized scoring, and formula-safe result
exports. State that it does not create investigations automatically.

## 3:45–4:30 — Monitoring

Open **Model Monitoring**. Explain that its reports compare train-reference distributions with
validation data and use heuristic drift thresholds. Emphasize that this is historical offline
monitoring, not production telemetry.

## 4:30–5:00 — Final evaluation

Open **Final Model Evaluation**. Show both models at validation-selected fixed thresholds, the
confusion matrices, ROC/PR curves, and the held-out test disclaimer. Close with limitations and
future production controls.
