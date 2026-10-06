from __future__ import annotations

import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
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
from sklearn.pipeline import FeatureUnion, Pipeline

from app.training_data import TARGET_EXAMPLES_PER_CLASS, augment_dataset


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sentiment_es.jsonl"
CHALLENGE_PATH = ROOT / "data" / "challenge_es.jsonl"
# Vercel Functions solo permite escritura persistente en /tmp durante la ejecución.
MODEL_PATH = (
    Path(tempfile.gettempdir()) / "sentiment_model.joblib"
    if os.getenv("VERCEL")
    else ROOT / "models" / "sentiment_model.joblib"
)
RANDOM_STATE = 42
MODEL_VERSION = 5

# Palabras de opinión generales. Las emociones se cubren además por familias
# morfológicas para reconocer género, número y derivados como ansiedad/ansioso.
POSITIVE_WORDS = {
    "adoro", "afortunado", "agradable", "alegre", "alegría", "amable", "amo",
    "apoyo", "aprecio", "ayuda", "ayudaron", "ayudó", "bien", "bonito",
    "brillante", "buena", "buenas", "bueno", "buenos", "buenísimo", "contento",
    "encanta", "encantó", "excelente", "excelentes", "fantástica", "fantástico",
    "felicitaciones", "felices", "feliz", "genial", "gracias", "gran", "gusta",
    "gustó", "hermoso", "increíble", "interesante", "lindo", "maravilla",
    "maravillosa", "mejor", "perfecto", "recomendable", "recomiendo", "útil",
}
NEGATIVE_WORDS = {
    "caro", "desastre", "errores", "equivocada", "falla", "fallando", "grosera",
    "ignoren", "borraron", "borró", "dañado", "dañada", "inútil", "lenta",
    "lentas", "mal", "mala", "malas", "malo", "malos", "nefasta", "peor",
    "perder", "perdí", "perdimos", "perdió", "reclamos", "retraso", "roto",
}

# (raíz sin tilde, valencia, familia emocional, intensidad base). Se comparan
# prefijos de cuatro o más letras; así se reconocen variaciones no vistas.
EMOTION_PREFIXES = (
    ("feli", 1, "alegría", 1.00), ("alegr", 1, "alegría", 1.00),
    ("content", 1, "alegría", 0.95), ("entusiasm", 1, "entusiasmo", 1.00),
    ("emocionad", 1, "entusiasmo", 0.95), ("ilusion", 1, "entusiasmo", 0.95),
    ("motivad", 1, "entusiasmo", 0.90), ("tranquil", 1, "calma", 0.90),
    ("alivi", 1, "calma", 0.90), ("optim", 1, "esperanza", 0.95),
    ("esperanz", 1, "esperanza", 0.90), ("orgull", 1, "satisfacción", 1.00),
    ("satisfe", 1, "satisfacción", 1.00), ("encant", 1, "satisfacción", 1.00),
    ("agradec", 1, "gratitud", 0.90), ("confiad", 1, "confianza", 0.90),
    ("inspirad", 1, "entusiasmo", 0.85), ("animad", 1, "alegría", 0.85),
    ("amor", 1, "afecto", 1.00), ("ador", 1, "afecto", 1.00),
    ("ansios", -1, "ansiedad", 1.00), ("ansiedad", -1, "ansiedad", 1.00),
    ("nervios", -1, "ansiedad", 0.95), ("angusti", -1, "ansiedad", 1.05),
    ("preocup", -1, "preocupación", 0.95), ("estres", -1, "ansiedad", 0.95),
    ("insegur", -1, "miedo", 0.90), ("mied", -1, "miedo", 1.00),
    ("temor", -1, "miedo", 0.95), ("asust", -1, "miedo", 1.00),
    ("enoj", -1, "enojo", 1.00), ("enfad", -1, "enojo", 1.00),
    ("molest", -1, "enojo", 0.90), ("irrit", -1, "enojo", 0.95),
    ("rabia", -1, "enojo", 1.10), ("furios", -1, "enojo", 1.10),
    ("odi", -1, "enojo", 1.10), ("trist", -1, "tristeza", 1.00),
    ("decepcion", -1, "tristeza", 1.00), ("llor", -1, "tristeza", 1.00),
    ("dolor", -1, "tristeza", 1.00), ("frustr", -1, "frustración", 1.00),
    ("cansad", -1, "frustración", 0.80), ("agotad", -1, "frustración", 0.90),
    ("desesper", -1, "frustración", 1.05), ("confus", -1, "confusión", 0.80),
    ("aburr", -1, "desagrado", 0.80), ("asco", -1, "desagrado", 1.05),
    ("repugn", -1, "desagrado", 1.05), ("horribl", -1, "desagrado", 1.05),
    ("terribl", -1, "desagrado", 1.05), ("pesim", -1, "tristeza", 0.90),
)

OPINION_PREFIXES = (
    ("excelent", 1, 0.95), ("perfect", 1, 0.95), ("maravill", 1, 1.00),
    ("fantast", 1, 0.95), ("increibl", 1, 0.90), ("recomend", 1, 0.90),
    ("agradabl", 1, 0.85), ("amabl", 1, 0.85), ("hermos", 1, 0.85),
    ("bonit", 1, 0.75), ("brillant", 1, 0.90), ("util", 1, 0.85),
    ("clar", 1, 0.70), ("sencill", 1, 0.75), ("rapid", 1, 0.75),
    ("establ", 1, 0.75), ("solucion", 1, 0.80), ("resolv", 1, 0.80),
    ("recuper", 1, 0.75), ("mejor", 1, 0.80),
    ("fall", -1, 0.90), ("error", -1, 0.85), ("inutil", -1, 0.95),
    ("lent", -1, 0.80), ("retras", -1, 0.80), ("rot", -1, 0.95),
    ("congel", -1, 0.90), ("desorden", -1, 0.80), ("problem", -1, 0.80),
    ("perd", -1, 0.90), ("borr", -1, 0.90), ("groser", -1, 0.90),
    ("danad", -1, 0.90), ("complicad", -1, 0.70), ("nefasta", -1, 1.00),
)

NEGATIONS = {"no", "nunca", "jamás", "nada", "nadie", "ni", "tampoco"}
INTENSIFIERS = {
    "absolutamente": 1.45, "bastante": 1.25, "demasiado": 1.50,
    "extremadamente": 1.65, "increíblemente": 1.45, "muy": 1.40,
    "profundamente": 1.50, "realmente": 1.25, "sumamente": 1.55,
    "totalmente": 1.45,
}
DIMINISHERS = {"algo": 0.70, "apenas": 0.55, "ligeramente": 0.65, "poco": 0.60}
CONTRAST_MARKERS = (
    "a pesar de", "aun así", "aunque", "en cambio", "mientras que", "pero",
    "por otro lado", "sin embargo",
)
MIXED_CUES = (
    "ambas sensaciones", "aspectos buenos y malos", "cosas buenas y malas",
    "emociones mezcladas", "ninguna emoción domina", "en proporciones similares",
    "sentimientos encontrados",
)
NEUTRAL_CUES = MIXED_CUES + (
    "buena ni mala", "bueno ni malo", "impresión general es intermedia",
    "ni bueno ni malo", "nada especial",
    "sigo indeciso", "sin una opinión definitiva", "todavía no puedo decir",
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
CONTRAST_SPLIT_PATTERN = re.compile(
    r"\b(?:a pesar de|aun así|aunque|en cambio|mientras que|pero|por otro lado|sin embargo)\b",
    flags=re.IGNORECASE,
)


@dataclass
class EmotionalEvidence:
    positive: float
    negative: float
    emotions: dict[str, float]

    @property
    def total(self) -> float:
        return self.positive + self.negative

    @property
    def balance(self) -> float:
        dominant = max(self.positive, self.negative)
        return min(self.positive, self.negative) / dominant if dominant else 0.0


def normalize_text(text: str) -> str:
    """Normaliza ruido común de comentarios sin eliminar emojis útiles."""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[\w._-]+", " USUARIO ", text)
    text = re.sub(r"#([\wáéíóúüñ]+)", r" \1 ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_token(token: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", token)
        if unicodedata.category(char) != "Mn"
    )


def split_sentiment_segments(text: str) -> list[str]:
    """Divide párrafos en unidades que pueden expresar sentimientos distintos."""
    normalized = normalize_text(text)
    segments = [segment.strip(" ,-") for segment in SEGMENT_PATTERN.split(normalized)]
    segments = [segment for segment in segments if TOKEN_PATTERN.search(segment)]
    return segments[:30] or [normalized]


def _token_polarity(token: str) -> tuple[int, str | None, float]:
    folded = fold_token(token)
    # La raíz más larga gana para evitar coincidencias parciales ambiguas.
    matches = [entry for entry in EMOTION_PREFIXES if folded.startswith(entry[0])]
    if matches:
        _, valence, emotion, weight = max(matches, key=lambda entry: len(entry[0]))
        return valence, emotion, weight
    opinion_matches = [entry for entry in OPINION_PREFIXES if folded.startswith(entry[0])]
    if opinion_matches:
        _, valence, weight = max(opinion_matches, key=lambda entry: len(entry[0]))
        return valence, None, weight
    if token in POSITIVE_WORDS:
        return 1, None, 0.85
    if token in NEGATIVE_WORDS:
        return -1, None, 0.85
    return 0, None, 0.0


def _analyze_clause_evidence(text: str) -> EmotionalEvidence:
    """Extrae evidencia gradual con negación, intensificadores y familias léxicas."""
    tokens = EVIDENCE_TOKEN_PATTERN.findall(normalize_text(text))
    positive = 0.0
    negative = 0.0
    emotions: dict[str, float] = {}
    negation_scope = 0
    modifier = 1.0
    previous_was_polarity = False

    for token in tokens:
        if token in ".!?;,:":
            negation_scope = 0
            modifier = 1.0
            previous_was_polarity = False
            continue
        if token in NEGATIONS:
            # En «no perdió nada», nada completa la negación anterior y no debe
            # invertir también la emoción que aparezca después de la frase.
            negation_scope = 0 if token in {"nada", "nadie"} and previous_was_polarity else 3
            modifier = 1.0
            previous_was_polarity = False
            continue
        if token in INTENSIFIERS:
            modifier = INTENSIFIERS[token]
            if negation_scope:
                negation_scope -= 1
            previous_was_polarity = False
            continue
        if token in DIMINISHERS:
            modifier = DIMINISHERS[token]
            if negation_scope:
                negation_scope -= 1
            previous_was_polarity = False
            continue

        valence, emotion, base_weight = _token_polarity(token)
        if valence:
            effective_valence = -valence if negation_scope else valence
            weight = base_weight * modifier
            if effective_valence > 0:
                positive += weight
            else:
                negative += weight
            if emotion:
                emotions[emotion] = emotions.get(emotion, 0.0) + weight
            modifier = 1.0
            previous_was_polarity = True
        else:
            previous_was_polarity = False
        if negation_scope:
            negation_scope -= 1

    positive += sum(text.count(emoji) for emoji in ("😊", "😍", "👏", "❤️", "😄", "🥳"))
    negative += sum(text.count(emoji) for emoji in ("😡", "😞", "😢", "😠", "💔", "😰"))
    return EmotionalEvidence(positive=positive, negative=negative, emotions=emotions)


def analyze_emotional_evidence(text: str) -> EmotionalEvidence:
    """Da algo más de peso a la conclusión que sigue a conectores adversativos."""
    normalized = normalize_text(text)
    clauses = [part.strip(" ,-") for part in CONTRAST_SPLIT_PATTERN.split(normalized)]
    clauses = [clause for clause in clauses if TOKEN_PATTERN.search(clause)]
    if len(clauses) <= 1:
        return _analyze_clause_evidence(text)

    positive = 0.0
    negative = 0.0
    emotions: dict[str, float] = {}
    for index, clause in enumerate(clauses):
        weight = 0.75 if index == 0 else min(1.15 + 0.10 * (index - 1), 1.35)
        clause_evidence = _analyze_clause_evidence(clause)
        positive += clause_evidence.positive * weight
        negative += clause_evidence.negative * weight
        for emotion, value in clause_evidence.emotions.items():
            emotions[emotion] = emotions.get(emotion, 0.0) + value * weight
    return EmotionalEvidence(positive=positive, negative=negative, emotions=emotions)


def sentiment_evidence(text: str) -> tuple[float, float]:
    evidence = analyze_emotional_evidence(text)
    return evidence.positive, evidence.negative


def has_neutral_cue(text: str) -> bool:
    normalized = normalize_text(text)
    return any(cue in normalized for cue in NEUTRAL_CUES)


def has_mixed_cue(text: str) -> bool:
    normalized = normalize_text(text)
    return any(cue in normalized for cue in MIXED_CUES)


def detects_sarcasm(
    text: str, positive_evidence: float, negative_evidence: float
) -> bool:
    normalized = normalize_text(text)
    return negative_evidence > 0 and (
        bool(SARCASM_OPENING_PATTERN.search(normalized))
        or "gracias por nada" in normalized
        or (
            any(cue in normalized for cue in SARCASM_CUES)
            and positive_evidence > 0
        )
    )


def _apply_floor_to_mapping(
    scores: dict[str, float], label: str, floor: float
) -> dict[str, float]:
    if scores[label] >= floor:
        return scores
    others = [name for name in scores if name != label]
    other_total = sum(scores[name] for name in others)
    adjusted = dict(scores)
    for name in others:
        adjusted[name] = (1.0 - floor) * scores[name] / other_total
    adjusted[label] = floor
    return adjusted


def fuzzy_memberships(evidence: EmotionalEvidence, neutral_cue: bool) -> dict[str, float]:
    """Convierte evidencia continua en pertenencias difusas que suman uno."""
    if evidence.total == 0:
        scores = {"positivo": 0.10, "neutral": 0.80, "negativo": 0.10}
    elif evidence.positive > 0 and evidence.negative > 0:
        # Con balance 1.0, neutral llega a 0.86; con balance bajo domina un polo.
        neutral = 0.20 + 0.66 * (evidence.balance ** 1.7)
        neutral = min(max(neutral, 0.20), 0.86)
        polar_mass = 1.0 - neutral
        scores = {
            "positivo": polar_mass * evidence.positive / evidence.total,
            "neutral": neutral,
            "negativo": polar_mass * evidence.negative / evidence.total,
        }
    else:
        strength = 1.0 - math.exp(-evidence.total)
        neutral = 0.22 - 0.10 * strength
        opposite = 0.04
        dominant = 1.0 - neutral - opposite
        scores = {
            "positivo": dominant if evidence.positive else opposite,
            "neutral": neutral,
            "negativo": dominant if evidence.negative else opposite,
        }

    if neutral_cue:
        scores = _apply_floor_to_mapping(scores, "neutral", 0.84)
    return scores


def _apply_floor(row: np.ndarray, class_indexes: dict[str, int], label: str, floor: float) -> np.ndarray:
    target_index = class_indexes[label]
    if row[target_index] >= floor:
        return row
    other_indexes = [index for index in range(len(row)) if index != target_index]
    other_total = sum(row[index] for index in other_indexes)
    for index in other_indexes:
        row[index] = (1.0 - floor) * row[index] / other_total
    row[target_index] = floor
    return row


class SentimentLexiconFeatures(BaseEstimator, TransformerMixin):
    """Señales lingüísticas que complementan TF-IDF y generalizan a palabras nuevas."""

    def fit(self, x: list[str], y: list[str] | None = None) -> "SentimentLexiconFeatures":
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        rows: list[list[float]] = []
        for text in texts:
            normalized = normalize_text(text)
            tokens = TOKEN_PATTERN.findall(normalized)
            evidence = analyze_emotional_evidence(text)
            negations = sum(token in NEGATIONS for token in tokens)
            expressive = min(text.count("!") + sum(char in "😊😍👏😡😞😰" for char in text), 3)
            contrasts = sum(marker in normalized for marker in CONTRAST_MARKERS)
            rows.append(
                [
                    evidence.positive * 2.0,
                    evidence.negative * 2.0,
                    negations,
                    expressive * 0.5,
                    contrasts,
                    min(evidence.positive, evidence.negative) * 2.0,
                    evidence.balance * 2.0,
                    float(has_neutral_cue(text)) * 2.0,
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
        raise ValueError("El conjunto de datos es demasiado pequeño.")
    return texts, labels


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 3),
                    min_df=1,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "chars",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    min_df=1,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            ("sentiment_signals", SentimentLexiconFeatures()),
        ]
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=2.5,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _hybrid_probabilities(pipeline: Pipeline, text: str) -> tuple[np.ndarray, dict[str, Any]]:
    classes = pipeline.classes_
    class_indexes = {label: index for index, label in enumerate(classes)}
    segments = split_sentiment_segments(text)
    global_probabilities = pipeline.predict_proba([text])[0]
    segment_probabilities = pipeline.predict_proba(segments)
    segment_weights = np.array(
        [1.0 + min(analyze_emotional_evidence(segment).total, 4.0) * 0.45 for segment in segments]
    )
    paragraph_probabilities = np.average(segment_probabilities, axis=0, weights=segment_weights)
    semantic_row = global_probabilities * 0.35 + paragraph_probabilities * 0.65
    # Calibración conservadora antes de aplicar las reglas, para no deshacer sus pisos.
    semantic_row = np.power(np.clip(semantic_row, 1e-8, 1.0), 1.0 / 1.30)
    semantic_row /= semantic_row.sum()

    evidence = analyze_emotional_evidence(text)
    neutral_cue = has_neutral_cue(text)
    sarcasm_detected = detects_sarcasm(text, evidence.positive, evidence.negative)
    mixed_emotions = (
        (evidence.positive > 0 and evidence.negative > 0) or has_mixed_cue(text)
    ) and not sarcasm_detected
    fuzzy = fuzzy_memberships(evidence, neutral_cue)
    fuzzy_row = np.array([fuzzy[str(label)] for label in classes])

    if mixed_emotions:
        fuzzy_weight = 0.72
    elif evidence.total:
        fuzzy_weight = 0.50
    elif neutral_cue:
        fuzzy_weight = 0.70
    else:
        fuzzy_weight = 0.25
    row = semantic_row * (1.0 - fuzzy_weight) + fuzzy_row * fuzzy_weight

    # Regla difusa de equilibrio: dos polos de fuerza semejante no se fuerzan a
    # una clase extrema. Los modificadores permiten romper el empate naturalmente.
    if mixed_emotions and evidence.balance >= 0.60:
        neutral_floor = 0.66 + 0.18 * evidence.balance
        row = _apply_floor(row, class_indexes, "neutral", neutral_floor)
    if neutral_cue and not sarcasm_detected:
        row = _apply_floor(row, class_indexes, "neutral", 0.84)
    if sarcasm_detected:
        row = _apply_floor(row, class_indexes, "negativo", 0.76)

    row /= row.sum()
    details = {
        "mixed_emotions": mixed_emotions,
        "sarcasm_detected": sarcasm_detected,
        "segments_analyzed": len(segments),
        "emotions_detected": [
            name for name, _ in sorted(
                evidence.emotions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }
    return row, details


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    challenge_path: Path = CHALLENGE_PATH,
) -> dict[str, Any]:
    curated_texts, curated_labels = load_dataset(data_path)
    texts, labels, augmented_size = augment_dataset(curated_texts, curated_labels)
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    challenge_texts, challenge_labels = load_dataset(challenge_path)
    challenge_predictions = []
    for text in challenge_texts:
        row, _ = _hybrid_probabilities(pipeline, text)
        challenge_predictions.append(str(pipeline.classes_[int(np.argmax(row))]))
    report = classification_report(
        challenge_labels, challenge_predictions, output_dict=True, zero_division=0
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "TF-IDF + regresión logística + inferencia difusa contextual",
        "language": "es",
        "classes": sorted(set(labels)),
        "dataset_size": len(texts),
        "curated_dataset_size": len(curated_texts),
        "augmented_dataset_size": augmented_size,
        "examples_per_class": TARGET_EXAMPLES_PER_CLASS,
        "train_size": len(texts),
        "evaluation_set": challenge_path.name,
        "test_size": len(challenge_texts),
        "test_accuracy": round(float(accuracy_score(challenge_labels, challenge_predictions)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "random_state": RANDOM_STATE,
    }
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
        for text in texts:
            row, details = _hybrid_probabilities(self._pipeline, text)
            best_index = int(np.argmax(row))
            scores = {str(label): float(row[index]) for index, label in enumerate(classes)}
            results.append(
                {
                    "text": text,
                    "sentiment": str(classes[best_index]),
                    "confidence": round(float(row[best_index]), 4),
                    **details,
                    "probabilities": {
                        "positivo": round(scores["positivo"], 4),
                        "neutral": round(scores["neutral"], 4),
                        "negativo": round(scores["negativo"], 4),
                    },
                }
            )
        return results


model = SentimentModel()
