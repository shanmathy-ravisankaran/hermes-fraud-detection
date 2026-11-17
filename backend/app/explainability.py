"""SHAP explanations for the fitted Hermes XGBoost pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


class ExplainabilityError(Exception):
    """Raised when a model cannot produce a reliable local explanation."""


def transaction_to_features(transaction: Any) -> pd.DataFrame:
    """Build the exact raw feature frame expected by the fitted pipeline."""
    return transactions_to_features(pd.DataFrame([transaction.model_dump()]))


def transactions_to_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Vectorize the exact feature engineering shared by single and batch scoring."""
    amounts = transactions["transaction_amount"].astype(float)
    transaction_times = transactions["transaction_dt"].astype(int)
    return pd.DataFrame({
        "TransactionAmt": amounts,
        "TransactionAmtLog": np.log1p(amounts),
        "TransactionHour": (transaction_times // 3600) % 24,
        "TransactionDay": transaction_times // 86400,
        "ProductCD": transactions["product_cd"].astype(str),
        "card4": transactions["card4"].astype(str),
        "card6": transactions["card6"].astype(str),
        "P_emaildomain": transactions["email_domain"].astype(str),
    })


class ModelExplanationService:
    """Caches a TreeExplainer bound to the fitted classifier in a pipeline."""

    output_scale = "raw_margin_log_odds"

    def __init__(self, model: Any):
        try:
            self.preprocessor = model.named_steps["preprocessing"]
            self.classifier = model.named_steps["classifier"]
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        except (AttributeError, KeyError) as exc:
            raise ExplainabilityError(
                "Explainability requires a fitted pipeline with preprocessing and classifier steps."
            ) from exc

        if not hasattr(self.classifier, "get_booster"):
            raise ExplainabilityError(
                "Explainability is only available for an XGBoost classifier."
            )

        try:
            self.explainer = shap.TreeExplainer(self.classifier, model_output="raw")
        except Exception as exc:
            raise ExplainabilityError(
                "The fitted XGBoost classifier could not initialize a SHAP explainer."
            ) from exc

    def _original_feature_name(self, transformed_feature_name: str) -> str:
        for feature_name in self.preprocessor.feature_names_in_:
            if transformed_feature_name == f"numeric__{feature_name}":
                return str(feature_name)
            if transformed_feature_name.startswith(f"categorical__{feature_name}_"):
                return str(feature_name)
        return transformed_feature_name

    def _contribution(
        self,
        index: int,
        transformed_values: np.ndarray,
        shap_values: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "feature_name": self.feature_names[index],
            "original_feature": self._original_feature_name(self.feature_names[index]),
            "feature_value": float(transformed_values[index]),
            "shap_value": float(shap_values[index]),
        }

    def explain(self, transaction: Any, *, top_features: int = 5) -> dict[str, Any]:
        features = transaction_to_features(transaction)
        transformed_features = self.preprocessor.transform(features)
        shap_values = np.asarray(self.explainer.shap_values(transformed_features))
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        if shap_values.ndim != 1 or len(shap_values) != len(self.feature_names):
            raise ExplainabilityError("Unexpected SHAP value shape for the fitted classifier.")

        expected_value = float(np.asarray(self.explainer.expected_value).reshape(-1)[0])
        raw_output = float(
            self.classifier.predict(transformed_features, output_margin=True)[0]
        )
        reconstructed_output = expected_value + float(np.sum(shap_values))
        reconstruction_error = abs(raw_output - reconstructed_output)
        if not np.isclose(raw_output, reconstructed_output, rtol=1e-5, atol=1e-4):
            raise ExplainabilityError(
                "SHAP values could not reconstruct the model raw output within tolerance."
            )

        # A single transformed row is safe to densify and preserves one-hot feature values.
        if hasattr(transformed_features, "getrow"):
            transformed_values = transformed_features.getrow(0).toarray().ravel()
        else:
            transformed_values = np.asarray(transformed_features)[0]

        positive_indices = np.where(shap_values > 0)[0]
        negative_indices = np.where(shap_values < 0)[0]
        top_positive = positive_indices[np.argsort(shap_values[positive_indices])[::-1]][:top_features]
        top_negative = negative_indices[np.argsort(shap_values[negative_indices])][:top_features]

        model_score = float(self.classifier.predict_proba(transformed_features)[0, 1])
        return {
            "model_score": model_score,
            "expected_value": expected_value,
            "raw_model_output": raw_output,
            "reconstructed_raw_output": reconstructed_output,
            "reconstruction_error": reconstruction_error,
            "top_positive_contributions": [
                self._contribution(index, transformed_values, shap_values)
                for index in top_positive
            ],
            "top_negative_contributions": [
                self._contribution(index, transformed_values, shap_values)
                for index in top_negative
            ],
        }
