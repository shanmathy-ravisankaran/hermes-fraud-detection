import json

import httpx
import pytest

from dashboard.investigation_api import (
    InvestigationApiConnectionError,
    InvestigationApiResponseError,
    explain_transaction,
    get_investigation,
    get_investigation_explanation,
    get_investigation_history,
    get_investigations,
    score_batch_file,
    update_investigation,
)

API_URL = "http://127.0.0.1:8000"


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_get_investigations_returns_api_response():
    def handler(request):
        assert request.url.path == "/investigations"
        assert request.url.params["limit"] == "50"
        return httpx.Response(200, json={"total": 1, "investigations": [{"id": 1}]})

    with make_client(handler) as client:
        result = get_investigations(API_URL, limit=50, client=client)

    assert result["total"] == 1
    assert result["investigations"][0]["id"] == 1


def test_status_filter_is_sent_to_api():
    def handler(request):
        assert request.url.params["status"] == "under_review"
        return httpx.Response(200, json={"total": 0, "investigations": []})

    with make_client(handler) as client:
        result = get_investigations(
            API_URL,
            investigation_status="under_review",
            client=client,
        )

    assert result["investigations"] == []


def test_update_investigation_sends_changed_notes_only():
    def handler(request):
        assert request.method == "PATCH"
        assert request.url.path == "/investigations/7"
        assert json.loads(request.content) == {
            "status": "confirmed_fraud",
            "analyst_notes": "Confirmed",
        }
        return httpx.Response(200, json={"id": 7, "status": "confirmed_fraud"})

    with make_client(handler) as client:
        result = update_investigation(
            API_URL,
            7,
            "confirmed_fraud",
            analyst_notes="Confirmed",
            notes_changed=True,
            client=client,
        )

    assert result["status"] == "confirmed_fraud"


def test_invalid_status_response_is_exposed():
    def handler(request):
        return httpx.Response(422, text="Invalid status")

    with make_client(handler) as client, pytest.raises(
        InvestigationApiResponseError
    ) as exc_info:
        update_investigation(API_URL, 7, "invalid", client=client)

    assert exc_info.value.status_code == 422


def test_missing_investigation_response_is_exposed():
    def handler(request):
        return httpx.Response(404, text="Investigation not found")

    with make_client(handler) as client, pytest.raises(
        InvestigationApiResponseError
    ) as exc_info:
        get_investigation(API_URL, 999, client=client)

    assert exc_info.value.status_code == 404


def test_connection_error_is_exposed():
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    with make_client(handler) as client, pytest.raises(InvestigationApiConnectionError):
        get_investigations(API_URL, client=client)


def test_get_investigation_history_returns_api_response():
    def handler(request):
        assert request.url.path == "/investigations/7/history"
        return httpx.Response(200, json={"total": 1, "history": [{"id": 1}]})

    with make_client(handler) as client:
        result = get_investigation_history(API_URL, 7, client=client)

    assert result["history"] == [{"id": 1}]


def test_explain_transaction_sends_transaction_to_api():
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/explain"
        assert json.loads(request.content)["transaction_amount"] == 100
        return httpx.Response(200, json={"model_score": 0.1})

    with make_client(handler) as client:
        result = explain_transaction(
            API_URL,
            {"transaction_amount": 100},
            client=client,
        )

    assert result["model_score"] == 0.1


def test_get_investigation_explanation_returns_api_response():
    def handler(request):
        assert request.url.path == "/investigations/7/explanation"
        return httpx.Response(200, json={"model_score": 0.1})

    with make_client(handler) as client:
        result = get_investigation_explanation(API_URL, 7, client=client)

    assert result["model_score"] == 0.1


def test_score_batch_file_posts_csv_multipart_data():
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/predict/batch"
        assert b"transaction_amount" in request.content
        return httpx.Response(200, json={"summary": {"total_uploaded_rows": 1}})

    with make_client(handler) as client:
        result = score_batch_file(
            API_URL,
            "transactions.csv",
            b"transaction_amount,transaction_dt,product_cd\n100,1,W\n",
            client=client,
        )

    assert result["summary"]["total_uploaded_rows"] == 1
