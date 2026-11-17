"""Safe batch-result CSV exports for browser downloads."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Any

EXPORT_COLUMNS = [
    "source_row_id",
    "model_score",
    "decision_threshold",
    "is_flagged",
    "validation_status",
    "error_message",
]


def _safe_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def batch_results_to_csv(results: list[dict[str, Any]]) -> str:
    """Create formula-injection-safe CSV without source transaction attributes."""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in results:
        writer.writerow({
            column: _safe_cell(result.get(column, ""))
            for column in EXPORT_COLUMNS
        })
    return output.getvalue()
