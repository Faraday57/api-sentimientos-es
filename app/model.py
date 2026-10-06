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
MODEL_VERSION = 4

POSITIVE_WORDS = {
    "adoro", "afortunado", "agradable", "alegre", "alegría", "amable", "amo", "apoyo", "aprecio", "ayuda", "ayudaron", "ayudó", "bien", "bonito", "brillante",
    "buena", "buenas", "bueno", "buenos", "buenísimo", "contento", "encanta", "encantó", "excelente", "excelentes",
    "fantástica", "fantásticas", "fantástico", "fantásticos", "felicitaciones", "felices", "feliz", "genial", "gracias", "gran", "gusta",
    "gustó", "hermoso", "ilusionado", "increíble", "interesante", "lindo", "maravilla", "maravillosa", "mejor",
    "orgulloso", "perfecto", "recomendable", "recomiendo", "satisfecho", "sonreír", "tranquilo", "útil",
}
NEGATIVE_WORDS = {
    "aburrido", "cansada", "caro", "confuso", "decepcionado", "decepcionante", "desastre",
    "enoja", "enojado", "enojada", "enfada", "enfadado", "errores", "equivocada", "falla", "fallando", "frustra", "frustrado", "frustrante", "furioso", "grosera", "horrible", "ignoren",
    "borraron", "borró", "dañado", "dañada", "ira", "inútil", "lenta", "lentas", "mal", "mala", "malas", "malo", "malos", "molesto", "molestos", "nefasta", "odia", "odiaba", "odiamos", "odiar", "odié", "odio", "peor",
    "perder", "perdí", "perdimos", "perdió", "pésimo", "rabia", "reclamos", "retraso", "roto", "triste", "terrible", "vergüenza",
}
NEGATIONS = {"no", "nunca", "jamás", "nada", "nadie", "ni"}
INTENSIFIERS = {"demasiado", "extremadamente", "muy", "realmente", "sumamente", "totalmente"}
DIMINISHERS = {"algo", "apenas", "poco", "ligeramente"}
CONTRAST_MARKERS = (
    "a pesar de", "aun así", "aunque", "en cambio", "mientras que", "pero",
    "por otro lado", "sin embargo",
)
MIXED_CUES = (
    "aspectos buenos y malos", "cosas buenas y malas", "en proporciones similares",
    "sentimientos encontrados",
)
NEUTRAL_CUES = MIXED_CUES + (
    "impresión general es intermedia", "ni bueno ni malo", "nada especial",
    "sin una opinión definitiva", "todavía no puedo decir",
)
SARCASM_OPENING_PATTERN = re.compile(
    r"^(?:qué\s+)?(?:buenísimo|excelente|fantástico|genial|maravilla)[,!:]",
    flags=re.IGNORECASE,
)
SARCASM_CUES = ("como siempre", "gracias por nada", "justo lo que faltaba", "otra vez")
TOKEN_PATTERN = re.compile(r"[a-záéíóúüñ]+")
EVIDENCE_TOKEN_PATTERN = re.compile(r"[a-záéíóúüñ]+|[.!?;,:]")
SEGMENT_PATTERN = re.compile(
    r"(?:[.!?;:\n]+|\b(?:a pesar de|aun así|aunque|en cambio|mientras que|pero|por otro lado|sin embargo)\b)",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normaliza ruido común de comentarios sin eliminar emojis útiles."""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[\w._-]+", " USUARIO ", text)
    text = re.sub(r"#([\wáéíóúüñ]+)", r" \1 ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentiment_segments(text: str) -> list[str]:
    """Divide párrafos en unidades que pueden expresar sentimientos distintos."""
    normalized = normalize_text(text)
    segments = [segment.strip(" ,-") for segment in SEGMENT_PATTERN.split(normalized)]
    segments = [segment for segment in segments if TOKEN_PATTERN.search(segment)]
    return segments[:30] or [normalized]


def sentiment_evidence(text: str) -> tuple[float, float]:
    """Calcula evidencia positiva y negativa considerando negación e intensidad."""
    tokens = EVIDENCE_TOKEN_PATTERN.findall(normalize_text(text))
    positive = 0.0
    negative = 0.0
    negation_scope = 0
    modifier = 1.0

    for token in tokens:
        if token in ".!?;,:":
            negation_scope = 0
            modifier = 1.0
            continue
        if token in NEGATIONS:
            negation_scope = 3
            modifier = 1.0
            continue
        if token in INTENSIFIERS:
            modifier = 1.5
            if negation_scope:
                negation_scope -= 1
            continue
        if token in DIMINISHERS:
            modifier = 0.65
            if negation_scope:
                negation_scope -= 1
            continue

        polarity = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if polarity:
            if negation_scope:
                polarity *= -1
            if polarity > 0:
                positive += modifier
            else:
                negative += modifier
            modifier = 1.0
        if negation_scope:
            negation_scope -= 1

    positive += sum(text.count(emoji) for emoji in ("😊", "😍", "👏", "❤️", "😄"))
    negative += sum(text.count(emoji) for emoji in ("😡", "😞", "😢", "😠", "💔"))
    return positive, negative


def has_neutral_cue(text: str) -> bool:
    normalized = normalize_text(text)
    return any(cue in normalized for cue in NEUTRAL_CUES)


def has_mixed_cue(text: str) -> bool:
    normalized = normalize_text(text)
    return any(cue in normalized for cue in MIXED_CUES)


def detects_sarcasm(text: str, negative_evidence: float) -> bool:
    normalized = normalize_text(text)
    return negative_evidence > 0 and (
        bool(SARCASM_OPENING_PATTERN.search(normalized))
        or "gracias por nada" in normalized
        or (
            any(cue in normalized for cue in SARCASM_CUES)
            and any(word in normalized for word in POSITIVE_WORDS)
        )
    )


class SentimentLexiconFeatures(BaseEstimator, TransformerMixin):
    """Señales lingüísticas generales que complementan TF-IDF en pocos datos."""

    def fit(self, x: list[str], y: list[str] | None = None) -> "SentimentLexiconFeatures":
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        rows: list[list[float]] = []
        for text in texts:
            normalized = normalize_text(text)
            tokens = TOKEN_PATTERN.findall(normalized)
            positive, negative = sentiment_evidence(text)
            negations = sum(token in NEGATIONS for token in tokens)
            expressive = min(text.count("!") + sum(char in "😊😍👏😡😞" for char in text), 3)
            contrasts = sum(marker in normalized for marker in CONTRAST_MARKERS)
            mixed = min(positive, negative)
            rows.append(
                [
                    positive * 2.0,
                    negative * 2.0,
                    negations,
                    expressive * 0.5,
                    contrasts,
                    mixed * 1.5,
                ]
            )
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
        "model_version": MODEL_VERSION,
        "algorithm": "TF-IDF (palabras + caracteres), señales emocionales y regresión logística",
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
            if bundle.get("metadata", {}).get("model_version") != MODEL_VERSION:
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
        classes = self._pipeline.classes_
        results: list[dict[str, Any]] = []
        class_indexes = {label: index for index, label in enumerate(classes)}

        for text in texts:
            segments = split_sentiment_segments(text)
            global_probabilities = self._pipeline.predict_proba([text])[0]
            segment_probabilities = self._pipeline.predict_proba(segments)
            segment_weights = np.array(
                [1.0 + min(sum(sentiment_evidence(segment)), 3.0) for segment in segments]
            )
            paragraph_probabilities = np.average(
                segment_probabilities, axis=0, weights=segment_weights
            )
            row = global_probabilities * 0.4 + paragraph_probabilities * 0.6

            positive_evidence, negative_evidence = sentiment_evidence(text)
            total_evidence = positive_evidence + negative_evidence
            neutral_cue = has_neutral_cue(text)
            sarcasm_detected = detects_sarcasm(text, negative_evidence)
            mixed_emotions = (
                (positive_evidence > 0 and negative_evidence > 0) or has_mixed_cue(text)
            ) and not sarcasm_detected

            if total_evidence:
                evidence_row = np.full(len(classes), 0.06, dtype=float)
                positive_share = positive_evidence / total_evidence
                negative_share = negative_evidence / total_evidence
                balance = min(positive_share, negative_share) / max(
                    positive_share, negative_share
                ) if mixed_emotions else 0.0
                neutral_share = 0.12 if not mixed_emotions else 0.35 + 0.30 * balance
                polar_mass = 1.0 - neutral_share
                evidence_row[class_indexes["positivo"]] = polar_mass * positive_share
                evidence_row[class_indexes["negativo"]] = polar_mass * negative_share
                evidence_row[class_indexes["neutral"]] = neutral_share
                evidence_row /= evidence_row.sum()
                row = row * 0.72 + evidence_row * 0.28

                # Emociones opuestas de intensidad similar se interpretan como tono mixto/neutral.
                if mixed_emotions and balance >= 0.55:
                    neutral_index = class_indexes["neutral"]
                    neutral_floor = 0.52 + 0.18 * balance
                    if row[neutral_index] < neutral_floor:
                        remaining = 1.0 - neutral_floor
                        positive_index = class_indexes["positivo"]
                        negative_index = class_indexes["negativo"]
                        polar = np.array(
                            [row[positive_index], row[negative_index]], dtype=float
                        )
                        evidence_mix = np.array([positive_share, negative_share])
                        polar = polar * 0.5 + evidence_mix * 0.5
                        polar /= polar.sum()
                        row[positive_index] = remaining * polar[0]
                        row[negative_index] = remaining * polar[1]
                        row[neutral_index] = neutral_floor

            if neutral_cue and not sarcasm_detected:
                neutral_index = class_indexes["neutral"]
                neutral_floor = 0.64
                if row[neutral_index] < neutral_floor:
                    other_indexes = [
                        index for index in range(len(classes)) if index != neutral_index
                    ]
                    other_total = sum(row[index] for index in other_indexes)
                    for index in other_indexes:
                        row[index] = (1.0 - neutral_floor) * row[index] / other_total
                    row[neutral_index] = neutral_floor

            if sarcasm_detected:
                negative_index = class_indexes["negativo"]
                negative_floor = 0.72
                if row[negative_index] < negative_floor:
                    other_indexes = [
                        index for index in range(len(classes)) if index != negative_index
                    ]
                    other_total = sum(row[index] for index in other_indexes)
                    for index in other_indexes:
                        row[index] = (1.0 - negative_floor) * row[index] / other_total
                    row[negative_index] = negative_floor

            # Suaviza la sobreconfianza habitual de modelos entrenados con pocos ejemplos.
            row = np.power(np.clip(row, 1e-8, 1.0), 1.0 / 1.35)
            row /= row.sum()
            best_index = int(np.argmax(row))
            scores = {label: float(row[index]) for index, label in enumerate(classes)}
            results.append(
                {
                    "text": text,
                    "sentiment": str(classes[best_index]),
                    "confidence": round(float(row[best_index]), 4),
                    "mixed_emotions": mixed_emotions,
                    "sarcasm_detected": sarcasm_detected,
                    "segments_analyzed": len(segments),
                    "probabilities": {
                        "positivo": round(scores["positivo"], 4),
                        "neutral": round(scores["neutral"], 4),
                        "negativo": round(scores["negativo"], 4),
                    },
                }
            )
        return results


model = SentimentModel()
