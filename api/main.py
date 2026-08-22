from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status

from api.model import TriageModel
from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


MODEL_VERSION = "logreg_tfidf_v2"

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / f"{MODEL_VERSION}.joblib"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = TriageModel(MODEL_PATH)

    try:
        model.load()

        app.state.model = model
        app.state.prediction_counter = Counter()

    except Exception as error:
        app.state.model = model
        app.state.prediction_counter = Counter()

        print(f"Failed to load model: {error}")

    yield


app = FastAPI(
    title="Medical Triage API",
    description=(
        "API for real-time medical report triage. "
        "The model expects text in English."
    ),
    version="1.0.0",
    lifespan=lifespan,
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

    triage_level, probabilities = model.predict(
        payload.text
    )

    request.app.state.prediction_counter[
        triage_level
    ] += 1

    return {
        "triage_level": triage_level,
        "probabilities": probabilities,
        "model_version": MODEL_VERSION,
    }