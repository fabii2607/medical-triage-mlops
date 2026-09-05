import time
from collections import Counter as CollectionCounter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from api.model import TriageModel
from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

MODEL_VERSION = "logreg_tfidf_v2"
MODEL_FILENAME = "logreg_tfidf.joblib"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME

REQUEST_COUNT = Counter(
    "medical_triage_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "medical_triage_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "medical_triage_predictions_total",
    "Total number of predictions by triage level",
    ["triage_level"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = TriageModel(MODEL_PATH)

    try:
        model.load()
        app.state.model = model
        app.state.prediction_counter = CollectionCounter()

    except Exception as error:  # noqa: BLE001
        app.state.model = model
        app.state.prediction_counter = CollectionCounter()

        print(f"Failed to load model: {error}")

    yield


app = FastAPI(
    title="Medical Triage API",
    description=(
        "API for real-time medical report triage. The model expects text in English."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_metrics(request: Request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    if request.url.path != "/metrics":
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request):
    model: TriageModel = request.app.state.model

    if not model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
):
    model: TriageModel = request.app.state.model

    if not model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    triage_level, probabilities = model.predict(payload.text)

    request.app.state.prediction_counter[triage_level] += 1

    PREDICTION_COUNT.labels(
        triage_level=triage_level,
    ).inc()

    return {
        "triage_level": triage_level,
        "probabilities": probabilities,
        "model_version": MODEL_VERSION,
    }
