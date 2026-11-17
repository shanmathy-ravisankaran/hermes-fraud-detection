from io import StringIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import database
from backend.app import main as main_module
from backend.app.main import app
from ml.batch.reporting import batch_results_to_csv


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "hermes-test.db")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_csv():
    return (
        "source_row_id,transaction_amount,transaction_dt,product_cd,card4,card6,email_domain\n"
        "batch-1,100,10400000,W,visa,debit,gmail.com\n"
        "batch-2,5800,10403600,C,mastercard,credit,yahoo.com\n"
    )


def upload_csv(client, content, filename="transactions.csv"):
    return client.post(
        "/predict/batch",
        files={"file": (filename, content.encode(), "text/csv")},
    )


def test_valid_csv_upload_scores_multiple_transactions(client, valid_csv):
    response = upload_csv(client, valid_csv)

    assert response.status_code == 200
    result = response.json()
    assert result["summary"]["total_uploaded_rows"] == 2
    assert result["summary"]["successfully_scored_rows"] == 2
    assert [item["source_row_id"] for item in result["results"]] == ["batch-1", "batch-2"]
    assert all(item["validation_status"] == "scored" for item in result["results"])


def test_batch_scores_match_single_prediction(client, valid_csv):
    batch_result = upload_csv(client, valid_csv).json()["results"][0]
    single_result = client.post(
        "/predict",
        json={
            "transaction_amount": 100,
            "transaction_dt": 10400000,
            "product_cd": "W",
            "card4": "visa",
            "card6": "debit",
            "email_domain": "gmail.com",
        },
    ).json()

    assert batch_result["model_score"] == single_result["fraud_score"]
    assert batch_result["is_flagged"] == single_result["is_suspicious"]


def test_invalid_rows_are_isolated_and_preserve_order(client, valid_csv):
    csv_content = valid_csv + "batch-3,-1,10407200,W,visa,debit,gmail.com\n"

    response = upload_csv(client, csv_content)

    assert response.status_code == 200
    result = response.json()
    assert result["summary"]["invalid_rows"] == 1
    assert [item["source_row_id"] for item in result["results"]] == [
        "batch-1",
        "batch-2",
        "batch-3",
    ]
    assert result["results"][2]["validation_status"] == "invalid"
    assert result["results"][2]["model_score"] is None


def test_negative_timestamp_is_invalid(client):
    response = upload_csv(
        client,
        "transaction_amount,transaction_dt,product_cd\n100,-1,W\n",
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["validation_status"] == "invalid"


def test_missing_required_columns_returns_row_errors(client):
    response = upload_csv(client, "transaction_amount,product_cd\n100,W\n")

    assert response.status_code == 200
    result = response.json()
    assert result["summary"]["invalid_rows"] == 1
    assert "transaction_dt" in result["results"][0]["error_message"]


def test_empty_and_malformed_csv_are_rejected(client):
    assert upload_csv(client, "").status_code == 422
    assert upload_csv(client, "   ").status_code == 422
    assert upload_csv(client, 'transaction_amount,transaction_dt,product_cd\n"bad').status_code == 422


def test_invalid_numeric_and_malformed_categorical_values_are_isolated(client):
    response = upload_csv(
        client,
        (
            "transaction_amount,transaction_dt,product_cd\n"
            "not-a-number,10400000,W\n"
            f"100,10400000,{'x' * 256}\n"
        ),
    )

    assert response.status_code == 200
    assert response.json()["summary"]["invalid_rows"] == 2


def test_unsupported_upload_type_is_rejected(client):
    response = client.post(
        "/predict/batch",
        files={"file": ("transactions.txt", b"not a csv", "text/plain")},
    )

    assert response.status_code == 415


def test_excessive_row_and_upload_limits_are_rejected(client, monkeypatch, valid_csv):
    monkeypatch.setattr(main_module, "BATCH_MAX_ROWS", 1)
    assert upload_csv(client, valid_csv).status_code == 422

    monkeypatch.setattr(main_module, "BATCH_MAX_ROWS", 500)
    monkeypatch.setattr(main_module, "BATCH_MAX_UPLOAD_BYTES", 10)
    assert upload_csv(client, valid_csv).status_code == 413


def test_duplicate_identifiers_and_missing_optional_fields(client):
    response = upload_csv(
        client,
        (
            "source_row_id,transaction_amount,transaction_dt,product_cd\n"
            "duplicate,100,10400000,W\n"
            "duplicate,200,10403600,C\n"
            "unique,300,10407200,R\n"
        ),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["summary"]["invalid_rows"] == 2
    assert result["results"][2]["validation_status"] == "scored"


def test_batch_scoring_does_not_create_investigations(client, valid_csv):
    response = upload_csv(client, valid_csv)

    assert response.status_code == 200
    assert client.get("/investigations").json()["total"] == 0


def test_csv_export_is_formula_safe():
    exported = batch_results_to_csv([{
        "source_row_id": "=DANGEROUS()",
        "model_score": 0.5,
        "decision_threshold": 0.7593,
        "is_flagged": False,
        "validation_status": "invalid",
        "error_message": "=formula",
    }])
    parsed = pd.read_csv(StringIO(exported))

    assert parsed.loc[0, "source_row_id"] == "'=DANGEROUS()"
    assert parsed.loc[0, "error_message"] == "'=formula"
