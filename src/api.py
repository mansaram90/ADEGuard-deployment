"""FastAPI backend for ADEGuard inference."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from .adeguard_inference import InputValidationError, NERModel, SeverityModel
    from .config import C_MODEL_PATH, model_path
    from .request_logging import RequestLoggingMiddleware
    from .metrics import RequestMetrics, RequestMetricsMiddleware
    from .artifact_metadata import verify_artifact
except ImportError:
    from adeguard_inference import InputValidationError, NERModel, SeverityModel
    from config import C_MODEL_PATH, model_path
    from request_logging import RequestLoggingMiddleware
    from metrics import RequestMetrics, RequestMetricsMiddleware
    from artifact_metadata import verify_artifact


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class EntityResponse(BaseModel):
    ADE: list[str]
    DRUG: list[str]


class SeverityResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    text: str
    entities: EntityResponse
    severity: SeverityResponse


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool


class ModelInfoResponse(BaseModel):
    ner_model_path: str
    severity_model_path: str
    ner_artifact_type: str | None = None
    severity_artifact_type: str | None = None
    ner_version: str
    severity_version: str
    ner_manifest_sha256: str
    severity_manifest_sha256: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    ner_info = verify_artifact(model_path, expected_task="token_classification")
    severity_info = verify_artifact(C_MODEL_PATH, expected_task="sequence_classification")
    ner_model = NERModel.load(model_path)
    severity_model = SeverityModel.load(C_MODEL_PATH)
    # Publish readiness only after both packages verify and both models load.
    app.state.ner_model = ner_model
    app.state.severity_model = severity_model
    app.state.model_info = ModelInfoResponse(
        ner_model_path=str(model_path), severity_model_path=str(C_MODEL_PATH),
        ner_artifact_type=ner_info["artifact_type"], severity_artifact_type=severity_info["artifact_type"],
        ner_version=ner_info["version"], severity_version=severity_info["version"],
        ner_manifest_sha256=ner_info["manifest_sha256"],
        severity_manifest_sha256=severity_info["manifest_sha256"],
    )
    try:
        yield
    finally:
        del app.state.ner_model, app.state.severity_model, app.state.model_info


app = FastAPI(
    title="ADEGuard API",
    description="Inference API for ADE entity extraction and severity prediction.",
    version="0.1.0",
    lifespan=lifespan,
)

# The standalone local frontend runs on a different port from this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Request-ID"],
)
# Wrap CORS too, including preflight requests.
app.add_middleware(RequestLoggingMiddleware)
app.state.metrics = RequestMetrics()
app.add_middleware(RequestMetricsMiddleware, metrics=app.state.metrics)


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    return request.app.state.metrics.response()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    is_ready = hasattr(request.app.state, "ner_model") and hasattr(request.app.state, "severity_model")
    return ReadyResponse(ready=is_ready)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    if not hasattr(request.app.state, "model_info"):
        raise HTTPException(status_code=503, detail="Models are not loaded")
    return request.app.state.model_info


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, fastapi_request: Request) -> PredictResponse:
    if not hasattr(fastapi_request.app.state, "ner_model"):
        raise HTTPException(status_code=503, detail="NER model is not loaded")

    if not hasattr(fastapi_request.app.state, "severity_model"):
        raise HTTPException(status_code=503, detail="Severity model is not loaded")

    try:
        ner_prediction = fastapi_request.app.state.ner_model.predict_one(request.text)
        severity_prediction = fastapi_request.app.state.severity_model.predict(request.text)
    except InputValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictResponse(
        text=request.text,
        entities=EntityResponse(
            ADE=ner_prediction.ade,
            DRUG=ner_prediction.drug,
        ),
        severity=SeverityResponse(
            label=severity_prediction.label,
            confidence=severity_prediction.confidence,
            probabilities=severity_prediction.probabilities,
        ),
    )
