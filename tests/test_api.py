
import pytest
from fastapi.testclient import TestClient

from backend.app import database
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "hermes-test.db")

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def transaction():
    return {
        "transaction_amount": 5800,
        "transaction_dt": 10400000,
        "product_cd": "W",
        "card4": "visa",
        "card6": "debit",
        "email_domain": "gmail.com",
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "XGBoost"


def test_valid_fraud_prediction(client, transaction):
    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_score"] <= 1
    assert 0 <= result["risk_score"] <= 100
    assert isinstance(result["is_suspicious"], bool)
    assert result["model_name"] == "XGBoost"

    assert result["is_suspicious"] == (
        result["fraud_score"] >= result["threshold"]
    )


def test_negative_transaction_amount(client):
    transaction = {
        "transaction_amount": -500,
        "transaction_dt": 10400000,
        "product_cd": "W",
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


def test_missing_required_fields(client):
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_negative_transaction_time(client):
    transaction = {
        "transaction_amount": 100,
        "transaction_dt": -1,
        "product_cd": "W",
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


def test_create_investigation(client, transaction):
    response = client.post("/investigations", json=transaction)

    assert response.status_code == 201
    result = response.json()
    assert result["investigation_id"] == 1
    assert result["status"] == "pending"
    assert result["prediction"]["model_name"] == "XGBoost"


def test_list_investigations(client, transaction):
    client.post("/investigations", json=transaction)

    response = client.get("/investigations")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["investigations"][0]["transaction_amount"] == 5800


def test_get_investigation(client, transaction):
    created = client.post("/investigations", json=transaction).json()

    response = client.get(f"/investigations/{created['investigation_id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["investigation_id"]
    assert response.json()["status"] == "pending"


def test_get_missing_investigation_returns_404(client):
    response = client.get("/investigations/999")

    assert response.status_code == 404


@pytest.mark.parametrize("limit", [0, 501])
def test_invalid_investigation_limit(client, limit):
    response = client.get(f"/investigations?limit={limit}")

    assert response.status_code == 422


def test_predict_does_not_save_investigations(client, transaction):
    prediction_response = client.post("/predict", json=transaction)
    investigations_response = client.get("/investigations")

    assert prediction_response.status_code == 200
    assert investigations_response.status_code == 200
    assert investigations_response.json() == {
        "total": 0,
        "investigations": [],
    }


def test_update_investigation_status(client, transaction):
    created = client.post("/investigations", json=transaction).json()

    response = client.patch(
        f"/investigations/{created['investigation_id']}",
        json={"status": "under_review"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "under_review"
    assert result["fraud_score"] == created["prediction"]["fraud_score"]
    assert result["model_name"] == "XGBoost"


def test_invalid_investigation_status_returns_422(client, transaction):
    created = client.post("/investigations", json=transaction).json()

    response = client.patch(
        f"/investigations/{created['investigation_id']}",
        json={"status": "invalid"},
    )

    assert response.status_code == 422


def test_invalid_investigation_status_filter_returns_422(client):
    response = client.get("/investigations?status=invalid")

    assert response.status_code == 422


def test_investigation_update_rejects_prediction_changes(client, transaction):
    created = client.post("/investigations", json=transaction).json()

    response = client.patch(
        f"/investigations/{created['investigation_id']}",
        json={"status": "under_review", "fraud_score": 0.0},
    )

    assert response.status_code == 422


def test_update_missing_investigation_returns_404(client):
    response = client.patch(
        "/investigations/999",
        json={"status": "under_review"},
    )

    assert response.status_code == 404


def test_analyst_notes_are_persisted(client, transaction):
    created = client.post("/investigations", json=transaction).json()
    investigation_id = created["investigation_id"]

    response = client.patch(
        f"/investigations/{investigation_id}",
        json={
            "status": "under_review",
            "analyst_notes": "Review cardholder verification.",
        },
    )

    assert response.status_code == 200
    assert response.json()["analyst_notes"] == "Review cardholder verification."
    assert client.get(f"/investigations/{investigation_id}").json()["analyst_notes"] == (
        "Review cardholder verification."
    )


def test_analyst_notes_can_be_cleared(client, transaction):
    investigation_id = client.post("/investigations", json=transaction).json()[
        "investigation_id"
    ]
    client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "under_review", "analyst_notes": "Initial note"},
    )

    response = client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "under_review", "analyst_notes": ""},
    )

    assert response.status_code == 200
    assert response.json()["analyst_notes"] == ""


def test_analyst_notes_length_is_limited(client, transaction):
    investigation_id = client.post("/investigations", json=transaction).json()[
        "investigation_id"
    ]

    response = client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "under_review", "analyst_notes": "x" * 2001},
    )

    assert response.status_code == 422


def test_omitted_notes_remain_unchanged(client, transaction):
    investigation_id = client.post("/investigations", json=transaction).json()[
        "investigation_id"
    ]
    client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "under_review", "analyst_notes": "Keep this note"},
    )

    response = client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "confirmed_fraud"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "confirmed_fraud"
    assert response.json()["analyst_notes"] == "Keep this note"


def test_filter_investigations_by_status(client, transaction):
    first_id = client.post("/investigations", json=transaction).json()[
        "investigation_id"
    ]
    client.post("/investigations", json=transaction)
    client.patch(
        f"/investigations/{first_id}",
        json={"status": "under_review"},
    )

    response = client.get("/investigations?status=under_review&limit=50")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["investigations"][0]["id"] == first_id
    assert result["investigations"][0]["status"] == "under_review"


def test_investigation_history_tracks_sequential_changes(client, transaction):
    investigation_id = client.post("/investigations", json=transaction).json()[
        "investigation_id"
    ]
    client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "under_review", "analyst_notes": "Review started"},
    )
    client.patch(
        f"/investigations/{investigation_id}",
        json={"status": "confirmed_fraud", "analyst_notes": "Fraud confirmed"},
    )

    response = client.get(f"/investigations/{investigation_id}/history")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 2
    assert result["history"][0]["previous_status"] == "pending"
    assert result["history"][0]["new_status"] == "under_review"
    assert result["history"][0]["updated_notes"] == "Review started"
    assert result["history"][1]["previous_status"] == "under_review"
    assert result["history"][1]["new_status"] == "confirmed_fraud"
    assert result["history"][1]["previous_notes"] == "Review started"


def test_missing_investigation_history_returns_404(client):
    response = client.get("/investigations/999/history")

    assert response.status_code == 404


def test_valid_explanation_response(client, transaction):
    prediction = client.post("/predict", json=transaction).json()
    response = client.post("/explain", json=transaction)

    assert response.status_code == 200
    result = response.json()
    assert result["model_name"] == "XGBoost"
    assert result["model_score"] == prediction["fraud_score"]
    assert result["explanation_output_scale"] == "raw_margin_log_odds"
    assert result["is_suspicious"] == prediction["is_suspicious"]
    assert result["top_positive_contributions"]
    assert result["top_negative_contributions"]


def test_explanation_reconstructs_raw_model_output(client, transaction):
    response = client.post("/explain", json=transaction)

    assert response.status_code == 200
    result = response.json()
    assert abs(result["raw_model_output"] - result["reconstructed_raw_output"]) < 1e-4
    assert result["reconstruction_error"] < 1e-4


def test_explanation_supports_missing_optional_fields(client):
    response = client.post(
        "/explain",
        json={
            "transaction_amount": 100,
            "transaction_dt": 10400000,
            "product_cd": "W",
        },
    )

    assert response.status_code == 200
    assert response.json()["model_name"] == "XGBoost"


def test_explanation_rejects_invalid_transaction(client):
    response = client.post(
        "/explain",
        json={
            "transaction_amount": -1,
            "transaction_dt": 10400000,
            "product_cd": "W",
        },
    )

    assert response.status_code == 422


def test_missing_investigation_explanation_returns_404(client):
    response = client.get("/investigations/999/explanation")

    assert response.status_code == 404


def test_investigation_explanation_does_not_mutate_record_or_history(client, transaction):
    investigation_id = client.post("/investigations", json=transaction).json()[
        "investigation_id"
    ]
    before_record = client.get(f"/investigations/{investigation_id}").json()
    before_history = client.get(f"/investigations/{investigation_id}/history").json()

    response = client.get(f"/investigations/{investigation_id}/explanation")

    assert response.status_code == 200
    assert client.get(f"/investigations/{investigation_id}").json() == before_record
    assert client.get(f"/investigations/{investigation_id}/history").json() == before_history
