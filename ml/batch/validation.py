"""Row-isolated validation for batch scoring uploads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import ValidationError

from backend.app.schemas import TransactionRequest

SOURCE_ROW_ID_COLUMN = "source_row_id"
REQUIRED_COLUMNS = ["transaction_amount", "transaction_dt", "product_cd"]
OPTIONAL_COLUMNS = ["card4", "card6", "email_domain"]
ACCEPTED_COLUMNS = [SOURCE_ROW_ID_COLUMN, *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
MAX_CATEGORICAL_LENGTH = 255


@dataclass
class ValidatedBatch:
    valid_rows: list[dict[str, Any]]
    results: list[dict[str, Any]]
    unexpected_columns: list[str]


def _is_missing(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _source_row_identifier(value: Any, row_number: int) -> str:
    if _is_missing(value) or not str(value).strip():
        return f"row-{row_number}"
    return str(value).strip()


def _format_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )


def validate_batch(data: pd.DataFrame) -> ValidatedBatch:
    """Validate every input row while retaining input order and invalid rows."""
    if data.empty:
        raise ValueError("The uploaded CSV contains no transaction rows.")

    unexpected_columns = [column for column in data.columns if column not in ACCEPTED_COLUMNS]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    source_values = (
        data[SOURCE_ROW_ID_COLUMN]
        if SOURCE_ROW_ID_COLUMN in data.columns
        else pd.Series([None] * len(data), index=data.index)
    )
    source_ids = [
        _source_row_identifier(value, row_number)
        for row_number, value in enumerate(source_values, start=1)
    ]
    duplicate_ids = {
        source_id
        for source_id in source_ids
        if source_ids.count(source_id) > 1
    }

    valid_rows = []
    results = []
    for input_order, (_, row) in enumerate(data.iterrows()):
        row_number = input_order + 1
        source_row_id = source_ids[input_order]
        errors = []
        if missing_columns:
            errors.append("Missing required columns: " + ", ".join(missing_columns))
        if source_row_id in duplicate_ids:
            errors.append(f"Duplicate source_row_id: {source_row_id}")

        payload = {}
        for column in REQUIRED_COLUMNS:
            payload[column] = row.get(column)
        for column in OPTIONAL_COLUMNS:
            value = row.get(column, "missing")
            payload[column] = "missing" if _is_missing(value) else value

        for column in ["product_cd", *OPTIONAL_COLUMNS]:
            value = payload[column]
            if isinstance(value, str) and len(value) > MAX_CATEGORICAL_LENGTH:
                errors.append(f"{column}: value exceeds {MAX_CATEGORICAL_LENGTH} characters")

        transaction = None
        if not errors:
            try:
                transaction = TransactionRequest.model_validate(payload)
            except ValidationError as exc:
                errors.append(_format_validation_error(exc))

        base_result = {
            "source_row_id": source_row_id,
            "source_row_number": row_number,
            "model_score": None,
            "decision_threshold": None,
            "is_flagged": None,
            "validation_status": "invalid" if errors else "valid",
            "error_message": "; ".join(errors),
        }
        results.append(base_result)
        if transaction is not None:
            valid_rows.append({
                "input_order": input_order,
                "transaction": transaction,
            })

    return ValidatedBatch(
        valid_rows=valid_rows,
        results=results,
        unexpected_columns=unexpected_columns,
    )
