from __future__ import annotations

from collections import defaultdict


TARGET_EXAMPLES_PER_CLASS = 500

POSITIVE_STATES = (
    "feliz", "alegre", "contento", "tranquilo", "aliviado", "optimista",
    "orgulloso", "entusiasmado", "satisfecho", "motivado", "esperanzado",
    "encantado", "agradecido", "confiado", "cómodo", "inspirado",
    "animado", "ilusionado", "emocionado", "afortunado",
)

NEGATIVE_STATES = (
    "ansioso", "enojado", "triste", "preocupado", "frustrado", "estresado",
    "nervioso", "angustiado", "decepcionado", "molesto", "irritado",
    "cansado", "agotado", "confundido", "aburrido", "inseguro",
    "desesperado", "furioso", "asustado", "pesimista",
)

CONTEXTS = (
    "con la publicación", "con el servicio", "con el resultado",
    "con la atención recibida", "después de probar la aplicación",
    "al leer los comentarios", "con la experiencia completa",
    "por lo que ocurrió hoy", "con el trabajo del equipo",
    "al terminar el proceso",
)

POSITIVE_TEMPLATES = (
    "Me siento {emotion} {context}; la experiencia ha sido muy buena.",
    "Estoy {emotion} {context} y valoro mucho el resultado.",
    "La verdad, quedé {emotion} {context}. Todo funcionó como esperaba.",
    "Aunque tenía dudas al principio, ahora estoy {emotion} {context}.",
    "Después de revisar cada detalle me siento {emotion} {context}.",
    "Hay un pequeño detalle por mejorar, pero sigo {emotion} {context}.",
)

NEGATIVE_TEMPLATES = (
    "Me siento {emotion} {context}; la experiencia ha sido muy mala.",
    "Estoy {emotion} {context} y el problema todavía no se resolvió.",
    "La verdad, quedé {emotion} {context}. Nada salió como esperaba.",
    "Aunque hubo algo rescatable, sigo {emotion} {context}.",
    "Después de revisar cada detalle me siento {emotion} {context}.",
    "Prometía ser bueno, pero terminé {emotion} {context}.",
)

MIXED_TEMPLATES = (
    "Me siento {positive} y {negative} {context}.",
    "Estoy {positive}, aunque también {negative} {context}.",
    "Una parte me deja {positive} y otra me hace sentir {negative} {context}.",
    "Tengo emociones mezcladas: estoy {positive} pero a la vez {negative} {context}.",
    "Por momentos me siento {positive}; por otros, {negative} {context}.",
)

FACTUAL_SUBJECTS = (
    "La reunión", "La publicación", "El formulario", "La clase", "El evento",
    "La entrega", "La encuesta", "La transmisión", "El registro", "La actividad",
)

FACTUAL_ACTIONS = (
    "comienza a las ocho", "termina mañana", "tiene tres secciones",
    "fue publicada esta mañana", "incluye un archivo adjunto",
    "se realizará de manera virtual", "está disponible en la plataforma",
    "requiere completar dos campos", "durará aproximadamente una hora",
    "cambió de fecha",
)


def _synthetic_examples() -> dict[str, list[str]]:
    generated: dict[str, list[str]] = defaultdict(list)

    for template in POSITIVE_TEMPLATES:
        for context in CONTEXTS:
            for state in POSITIVE_STATES:
                generated["positivo"].append(template.format(emotion=state, context=context))

    for template in NEGATIVE_TEMPLATES:
        for context in CONTEXTS:
            for state in NEGATIVE_STATES:
                generated["negativo"].append(template.format(emotion=state, context=context))

    # Pares balanceados: ambos polos están presentes y ninguno domina al otro.
    for positive_index, positive in enumerate(POSITIVE_STATES):
        for negative_index, negative in enumerate(NEGATIVE_STATES):
            context = CONTEXTS[(positive_index + negative_index) % len(CONTEXTS)]
            template = MIXED_TEMPLATES[(positive_index * 3 + negative_index) % len(MIXED_TEMPLATES)]
            generated["neutral"].append(
                template.format(positive=positive, negative=negative, context=context)
            )

    for subject in FACTUAL_SUBJECTS:
        for action in FACTUAL_ACTIONS:
            generated["neutral"].append(f"{subject} {action}.")

    return generated


def augment_dataset(
    curated_texts: list[str],
    curated_labels: list[str],
    target_per_class: int = TARGET_EXAMPLES_PER_CLASS,
) -> tuple[list[str], list[str], int]:
    """Amplía de forma reproducible el corpus y conserva exactamente el balance."""
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for text, label in zip(curated_texts, curated_labels):
        key = " ".join(text.lower().split())
        if key not in seen[label]:
            grouped[label].append(text)
            seen[label].add(key)

    generated = _synthetic_examples()
    for label in ("positivo", "neutral", "negativo"):
        for text in generated[label]:
            if len(grouped[label]) >= target_per_class:
                break
            key = " ".join(text.lower().split())
            if key not in seen[label]:
                grouped[label].append(text)
                seen[label].add(key)
        if len(grouped[label]) < target_per_class:
            raise ValueError(f"No hay suficientes ejemplos sintéticos para {label}.")

    texts: list[str] = []
    labels: list[str] = []
    # Intercalado por clase para que el orden del corpus no introduzca sesgos.
    for index in range(target_per_class):
        for label in ("positivo", "neutral", "negativo"):
            texts.append(grouped[label][index])
            labels.append(label)
    return texts, labels, len(texts) - len(curated_texts)
