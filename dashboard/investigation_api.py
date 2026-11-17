"""HTTP client helpers for the local Hermes investigation API."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT = 10.0


class InvestigationApiError(Exception):
    """Base exception for investigation API communication errors."""


class InvestigationApiConnectionError(InvestigationApiError):
    """Raised when the local API cannot be reached."""


class InvestigationApiResponseError(InvestigationApiError):
    """Raised when the API returns a non-success response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API returned {status_code}: {detail}")


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _request(
    method: str,
    base_url: str,
    path: str,
    *,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    owns_client = client is None
    request_client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        response = request_client.request(
            method,
            _api_url(base_url, path),
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise InvestigationApiResponseError(
            exc.response.status_code,
            exc.response.text[:500],
        ) from exc
    except httpx.RequestError as exc:
        raise InvestigationApiConnectionError(str(exc)) from exc
    finally:
        if owns_client:
            request_client.close()


def get_investigations(
    base_url: str,
    *,
    limit: int = 100,
    investigation_status: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if investigation_status is not None:
        params["status"] = investigation_status

    return _request(
        "GET",
        base_url,
        "/investigations",
        client=client,
        params=params,
    )


def get_investigation(
    base_url: str,
    investigation_id: int,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    return _request(
        "GET",
        base_url,
        f"/investigations/{investigation_id}",
        client=client,
    )


def update_investigation(
    base_url: str,
    investigation_id: int,
    investigation_status: str,
    *,
    analyst_notes: str | None = None,
    notes_changed: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": investigation_status}
    if notes_changed:
        payload["analyst_notes"] = analyst_notes

    return _request(
        "PATCH",
        base_url,
        f"/investigations/{investigation_id}",
        client=client,
        json=payload,
    )


def get_investigation_history(
    base_url: str,
    investigation_id: int,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    return _request(
        "GET",
        base_url,
        f"/investigations/{investigation_id}/history",
        client=client,
    )


def explain_transaction(
    base_url: str,
    transaction: dict[str, Any],
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        base_url,
        "/explain",
        client=client,
        json=transaction,
    )


def get_investigation_explanation(
    base_url: str,
    investigation_id: int,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    return _request(
        "GET",
        base_url,
        f"/investigations/{investigation_id}/explanation",
        client=client,
    )


def score_batch_file(
    base_url: str,
    filename: str,
    content: bytes,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    return _request(
        "POST",
        base_url,
        "/predict/batch",
        client=client,
        files={"file": (filename, content, "text/csv")},
    )
