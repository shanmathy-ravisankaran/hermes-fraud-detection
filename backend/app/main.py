import json
import os
from contextlib import asynccontextmanager
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.database import (
    get_investigation,
    initialize_database,
    list_investigation_history,
    list_investigations,
    save_investigation,
    update_investigation,
)
from backend.app.explainability import (
    ExplainabilityError,
    ModelExplanationService,
    transaction_to_features,
)
from backend.app.schemas import TransactionRequest
from ml.batch.scorer import score_batch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    MODEL_CONFIG = json.load(file)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / MODEL_CONFIG["model_file"]
)

FRAUD_THRESHOLD = float(
    MODEL_CONFIG["decision_threshold"]
)
BATCH_MAX_UPLOAD_BYTES = int(os.getenv("HERMES_BATCH_MAX_UPLOAD_BYTES", "1048576"))
BATCH_MAX_ROWS = int(os.getenv("HERMES_BATCH_MAX_ROWS", "500"))


class PredictionResponse(BaseModel):
    fraud_score: float
    risk_score: float
    threshold: float
    is_suspicious: bool
    recommendation: str
    model_name: str


class InvestigationCreateResponse(BaseModel):
    investigation_id: int
    status: str
    prediction: PredictionResponse


class InvestigationStatus(str, Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"


class InvestigationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: InvestigationStatus
    analyst_notes: str | None = Field(default=None, max_length=2000)


class FeatureContribution(BaseModel):
    feature_name: str
    original_feature: str
    feature_value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    model_name: str
    model_score: float
    threshold: float
    is_suspicious: bool
    explanation_output_scale: str
    expected_value: float
    raw_model_output: float
    reconstructed_raw_output: float
    reconstruction_error: float
    top_positive_contributions: list[FeatureContribution]
    top_negative_contributions: list[FeatureContribution]


def create_prediction(
    transaction: TransactionRequest,
    request: Request,
) -> PredictionResponse:
    features = transaction_to_features(transaction)

    fraud_score = float(request.app.state.model.predict_proba(features)[0, 1])
    suspicious = fraud_score >= FRAUD_THRESHOLD

    return PredictionResponse(
        fraud_score=round(fraud_score, 4),
        risk_score=round(fraud_score * 100, 2),
        threshold=FRAUD_THRESHOLD,
        is_suspicious=suspicious,
        recommendation=(
            "Manual review recommended"
            if suspicious
            else "No alert at current threshold"
        ),
        model_name="XGBoost",
    )


def get_explanation_service(request: Request) -> ModelExplanationService:
    service = getattr(request.app.state, "explanation_service", None)
    if service is not None:
        return service

    try:
        service = ModelExplanationService(request.app.state.model)
    except ExplainabilityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model explainability is unavailable for the loaded model.",
        ) from exc

    request.app.state.explanation_service = service
    return service


def create_explanation(
    transaction: TransactionRequest,
    request: Request,
) -> ExplanationResponse:
    try:
        explanation = get_explanation_service(request).explain(transaction)
    except ExplainabilityError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate a model explanation.",
        ) from exc

    model_score = explanation["model_score"]
    return ExplanationResponse(
        model_name="XGBoost",
        model_score=round(model_score, 4),
        threshold=FRAUD_THRESHOLD,
        is_suspicious=model_score >= FRAUD_THRESHOLD,
        explanation_output_scale=ModelExplanationService.output_scale,
        expected_value=explanation["expected_value"],
        raw_model_output=explanation["raw_model_output"],
        reconstructed_raw_output=explanation["reconstructed_raw_output"],
        reconstruction_error=explanation["reconstruction_error"],
        top_positive_contributions=explanation["top_positive_contributions"],
        top_negative_contributions=explanation["top_negative_contributions"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Trained model not found: {MODEL_PATH}"
        )

    app.state.model = joblib.load(MODEL_PATH)

    initialize_database()

    yield


app = FastAPI(
    title="Hermes Fraud Detection API",
    description="Machine learning fraud risk scoring service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, request: Request):
    try:
        return create_prediction(transaction, request)

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/explain", response_model=ExplanationResponse)
def explain(transaction: TransactionRequest, request: Request):
    return create_explanation(transaction, request)


@app.post(
    "/predict/batch",
    summary="Score a CSV batch of transactions without saving investigations",
    description=(
        "Upload a UTF-8 CSV with required columns transaction_amount, transaction_dt, and "
        "product_cd. Optional columns are card4, card6, email_domain, and source_row_id. "
        "Valid rows are scored together; invalid rows are returned with row-level errors."
    ),
)
async def predict_batch(request: Request, file: Annotated[UploadFile, File(...)]):
    filename = file.filename or ""
    supported_content_types = {"text/csv", "application/csv", "application/vnd.ms-excel"}
    if not filename.lower().endswith(".csv") or (
        file.content_type and file.content_type not in supported_content_types
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a CSV file with a .csv extension.",
        )

    content = await file.read(BATCH_MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    if len(content) > BATCH_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds the {BATCH_MAX_UPLOAD_BYTES}-byte local development limit.",
        )

    try:
        csv_text = content.decode("utf-8-sig")
        transactions = pd.read_csv(StringIO(csv_text), on_bad_lines="error")
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid UTF-8 CSV.") from exc

    if transactions.empty:
        raise HTTPException(status_code=422, detail="The uploaded CSV contains no transaction rows.")
    if len(transactions) > BATCH_MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"CSV exceeds the {BATCH_MAX_ROWS}-row local development limit.",
        )

    try:
        return score_batch(
            request.app.state.model,
            transactions,
            threshold=FRAUD_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/investigations",
    response_model=InvestigationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_investigation(transaction: TransactionRequest, request: Request):
    try:
        prediction = create_prediction(transaction, request)
        investigation_id = save_investigation(transaction, prediction)
        return InvestigationCreateResponse(
            investigation_id=investigation_id,
            status="pending",
            prediction=prediction,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/investigations")
def read_investigations(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    investigation_status: Annotated[
        InvestigationStatus | None,
        Query(alias="status"),
    ] = None,
):
    investigations = list_investigations(
        limit=limit,
        investigation_status=(
            investigation_status.value if investigation_status is not None else None
        ),
    )
    return {
        "total": len(investigations),
        "investigations": investigations,
    }


@app.patch("/investigations/{investigation_id}")
def update_investigation_record(
    investigation_id: int,
    update: InvestigationUpdateRequest,
):
    updated_investigation = update_investigation(
        investigation_id=investigation_id,
        investigation_status=update.status.value,
        analyst_notes=update.analyst_notes,
        notes_provided="analyst_notes" in update.model_fields_set,
    )
    if updated_investigation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )
    return updated_investigation


@app.get("/investigations/{investigation_id}/history")
def read_investigation_history(investigation_id: int):
    if get_investigation(investigation_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )
    history = list_investigation_history(investigation_id)
    return {
        "total": len(history),
        "history": history,
    }


@app.get(
    "/investigations/{investigation_id}/explanation",
    response_model=ExplanationResponse,
)
def read_investigation_explanation(investigation_id: int, request: Request):
    investigation = get_investigation(investigation_id)
    if investigation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )

    transaction = TransactionRequest(
        transaction_amount=investigation["transaction_amount"],
        transaction_dt=investigation["transaction_dt"],
        product_cd=investigation["product_cd"],
        card4=investigation["card4"] or "missing",
        card6=investigation["card6"] or "missing",
        email_domain=investigation["email_domain"] or "missing",
    )
    return create_explanation(transaction, request)


@app.get("/investigations/{investigation_id}")
def read_investigation(investigation_id: int):
    investigation = get_investigation(investigation_id)
    if investigation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )
    return investigation
