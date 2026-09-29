from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.model import model
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="API de Análisis de Sentimientos",
    version="1.0.0",
    description=(
        "Clasifica comentarios en español como positivos, neutrales o negativos. "
        "Proyecto académico de Inteligencia Artificial y Machine Learning."
    ),
    lifespan=lifespan,
    contact={"name": "Proyecto académico"},
)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Sistema"])
async def health() -> dict:
    return {"status": "ok" if model.ready else "not_ready", "model_loaded": model.ready}


@app.get("/api/v1/model/info", tags=["Modelo"])
async def model_info() -> dict:
    return model.metadata


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Predicción"],
    summary="Analizar un comentario",
)
async def predict(payload: PredictionRequest) -> dict:
    if not model.ready:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return model.predict([payload.text])[0]


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predicción"],
    summary="Analizar varios comentarios",
)
async def predict_batch(payload: BatchPredictionRequest) -> dict:
    if not model.ready:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    predictions = model.predict(payload.texts)
    return {"predictions": predictions, "count": len(predictions)}

