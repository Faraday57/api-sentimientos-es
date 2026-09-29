from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sentiment_es.jsonl"
# Vercel Functions solo permite escritura persistente en /tmp durante la ejecución.
MODEL_PATH = (
    Path(tempfile.gettempdir()) / "sentiment_model.joblib"
    if os.getenv("VERCEL")
    else ROOT / "models" / "sentiment_model.joblib"
)
RANDOM_STATE = 42

POSITIVE_WORDS = {
    "amo", "alegría", "amable", "apoyo", "aprecio", "bien", "bonito", "brillante",
    "buena", "bueno", "buenísimo", "contento", "encanta", "encantó", "excelente",
    "fantástica", "fantástico", "felicitaciones", "feliz", "genial", "gracias", "gran",
    "hermoso", "increíble", "interesante", "lindo", "maravilla", "maravillosa", "mejor",
    "perfecto", "recomendable", "recomiendo", "satisfecho", "sonreír", "útil",
}
NEGATIVE_WORDS = {
    "aburrido", "cansada", "caro", "confuso", "decepcionado", "decepcionante", "desastre",
    "errores", "equivocada", "fallando", "frustrante", "grosera", "horrible", "ignoren",
    "inútil", "lenta", "mal", "mala", "malo", "molesto", "nefasta", "odio", "peor",
    "perdí", "pésimo", "reclamos", "roto", "triste", "terrible", "vergüenza",
}
NEGATIONS = {"no", "nunca", "jamás", "nada", "nadie", "ni"}


def normalize_text(text: str) -> str:
    """Normaliza ruido común de comentarios sin eliminar emojis útiles."""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[\w._-]+", " USUARIO ", text)
    text = re.sub(r"#([\wáéíóúüñ]+)", r" \1 ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


class SentimentLexiconFeatures(BaseEstimator, TransformerMixin):
    """Señales lingüísticas generales que complementan TF-IDF en pocos datos."""

    def fit(self, x: list[str], y: list[str] | None = None) -> "SentimentLexiconFeatures":
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        rows: list[list[float]] = []
        for text in texts:
            normalized = normalize_text(text)
            tokens = re.findall(r"[a-záéíóúüñ]+", normalized)
            positive = sum(token in POSITIVE_WORDS for token in tokens)
            negative = sum(token in NEGATIVE_WORDS for token in tokens)
            negations = sum(token in NEGATIONS for token in tokens)
            expressive = min(text.count("!") + sum(char in "😊😍👏😡😞" for char in text), 3)
            rows.append([positive * 2.0, negative * 2.0, negations, expressive * 0.5])
        return csr_matrix(rows, dtype=float)


def load_dataset(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("label") not in {"positivo", "neutral", "negativo"}:
                raise ValueError(f"Etiqueta inválida en la línea {line_number}")
            texts.append(item["text"])
            labels.append(item["label"])
    if len(texts) < 30:
        raise ValueError("El conjunto de entrenamiento es demasiado pequeño.")
    return texts, labels


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=12_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "chars",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=18_000,
                    sublinear_tf=True,
                ),
            ),
            ("sentiment_signals", SentimentLexiconFeatures()),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=3.0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def train_and_save(
    data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH
) -> dict[str, Any]:
    texts, labels = load_dataset(data_path)
    train_x, test_x, train_y, test_y = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    pipeline = build_pipeline()
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    report = classification_report(
        test_y, predictions, output_dict=True, zero_division=0
    )
    metadata = {
        "algorithm": "TF-IDF (palabras + caracteres) y regresión logística",
        "language": "es",
        "classes": sorted(set(labels)),
        "dataset_size": len(texts),
        "train_size": len(train_x),
        "test_size": len(test_x),
        "test_accuracy": round(float(accuracy_score(test_y, predictions)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "random_state": RANDOM_STATE,
    }
    # Finalmente se reentrena con todos los ejemplos para servir el mejor modelo.
    pipeline.fit(texts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    return metadata


class SentimentModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if not self.model_path.exists():
                train_and_save(model_path=self.model_path)
            bundle = joblib.load(self.model_path)
            self._pipeline = bundle["pipeline"]
            self.metadata = bundle["metadata"]

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        if self._pipeline is None:
            raise RuntimeError("El modelo todavía no está cargado.")
        probabilities = self._pipeline.predict_proba(texts)
        classes = self._pipeline.classes_
        results: list[dict[str, Any]] = []
        for text, row in zip(texts, probabilities):
            best_index = int(np.argmax(row))
            scores = {label: float(row[index]) for index, label in enumerate(classes)}
            results.append(
                {
                    "text": text,
                    "sentiment": str(classes[best_index]),
                    "confidence": round(float(row[best_index]), 4),
                    "probabilities": {
                        "positivo": round(scores["positivo"], 4),
                        "neutral": round(scores["neutral"], 4),
                        "negativo": round(scores["negativo"], 4),
                    },
                }
            )
        return results


model = SentimentModel()
