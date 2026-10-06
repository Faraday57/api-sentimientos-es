from typing import Literal

from pydantic import BaseModel, Field, field_validator


SentimentLabel = Literal["positivo", "neutral", "negativo"]


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Comentario en español que se desea analizar.",
        examples=["Me encantó la publicación, muchas gracias por compartir."],
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El comentario no puede estar vacío.")
        return value


class ProbabilityScores(BaseModel):
    positivo: float
    neutral: float
    negativo: float


class PredictionResponse(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float
    mixed_emotions: bool
    sarcasm_detected: bool
    segments_analyzed: int
    emotions_detected: list[str]
    probabilities: ProbabilityScores


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        clean = [value.strip() for value in values]
        if any(not value for value in clean):
            raise ValueError("Ningún comentario puede estar vacío.")
        if any(len(value) > 5000 for value in clean):
            raise ValueError("Cada comentario debe tener como máximo 5000 caracteres.")
        return clean


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
