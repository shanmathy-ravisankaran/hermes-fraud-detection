"""Vectorized batch scoring using the fitted Hermes pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from backend.app.explainability import transactions_to_features
from ml.batch.validation import ValidatedBatch, validate_batch


def score_batch(model: Any, data: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
    """Validate rows independently, then score every valid row in one model call."""
    started_at = perf_counter()
    validated: ValidatedBatch = validate_batch(data)
    results = validated.results
    if validated.valid_rows:
        transactions = pd.DataFrame([
            item["transaction"].model_dump() for item in validated.valid_rows
        ])
        try:
            scores = model.predict_proba(transactions_to_features(transactions))[:, 1]
        except (TypeError, ValueError) as exc:
            for item in validated.valid_rows:
                result = results[item["input_order"]]
                result["validation_status"] = "unscored"
                result["error_message"] = f"Scoring failed: {exc}"
        else:
            for item, score in zip(validated.valid_rows, scores, strict=True):
                result = results[item["input_order"]]
                result.update({
                    "model_score": round(float(score), 4),
                    "decision_threshold": float(threshold),
                    "is_flagged": bool(score >= threshold),
                    "validation_status": "scored",
                })

    scored_results = [item for item in results if item["validation_status"] == "scored"]
    flagged_count = sum(item["is_flagged"] for item in scored_results)
    summary = {
        "total_uploaded_rows": len(data),
        "valid_rows": sum(item["validation_status"] != "invalid" for item in results),
        "invalid_rows": sum(item["validation_status"] == "invalid" for item in results),
        "successfully_scored_rows": len(scored_results),
        "flagged_transactions": flagged_count,
        "not_flagged_transactions": len(scored_results) - flagged_count,
        "flagged_percentage": (flagged_count / len(scored_results) * 100) if scored_results else 0.0,
        "average_model_score": (
            sum(item["model_score"] for item in scored_results) / len(scored_results)
            if scored_results
            else None
        ),
        "processing_duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "scores_are_uncalibrated": True,
    }
    return {
        "summary": summary,
        "results": results,
        "validation_errors": [item for item in results if item["validation_status"] != "scored"],
        "unexpected_columns_ignored": validated.unexpected_columns,
    }
