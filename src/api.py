"""FastAPI backend for ADEGuard inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .adeguard_inference import InputValidationError, NERModel, SeverityModel
    from .config import C_MODEL_PATH, model_path
    from .request_logging import RequestLoggingMiddleware
except ImportError:
    from adeguard_inference import InputValidationError, NERModel, SeverityModel
    from config import C_MODEL_PATH, model_path
    from request_logging import RequestLoggingMiddleware


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


def read_artifact_type(path: str) -> str | None:
    metadata_path = Path(path) / "ADEGUARD_DEMO_ARTIFACT.json"
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    return metadata.get("artifact_type")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ner_model = NERModel.load(model_path)
    app.state.severity_model = SeverityModel.load(C_MODEL_PATH)
    yield


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
# Added last so it wraps CORS too, including preflight requests.
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    is_ready = hasattr(request.app.state, "ner_model") and hasattr(request.app.state, "severity_model")
    return ReadyResponse(ready=is_ready)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        ner_model_path=str(model_path),
        severity_model_path=str(C_MODEL_PATH),
        ner_artifact_type=read_artifact_type(model_path),
        severity_artifact_type=read_artifact_type(C_MODEL_PATH),
    )


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
