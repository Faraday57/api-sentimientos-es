from fastapi.testclient import TestClient

from app.main import app


def test_health_and_model_info() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}
        info = client.get("/api/v1/model/info")
        assert info.status_code == 200
        metadata = info.json()
        assert metadata["model_version"] == 5
        assert metadata["dataset_size"] == 1500
        assert metadata["test_size"] == 45


def test_predict_positive_comment() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict", json={"text": "Excelente trabajo, me encantó muchísimo"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] == "positivo"
        assert 0 <= body["confidence"] <= 1
        assert abs(sum(body["probabilities"].values()) - 1) < 0.01


def test_predict_negative_comment() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict", json={"text": "Pésimo servicio, estoy muy decepcionado"}
        )
        assert response.status_code == 200
        assert response.json()["sentiment"] == "negativo"


def test_batch_and_validation() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["Muchas gracias", "La reunión es mañana", "Qué desastre"]},
        )
        assert response.status_code == 200
        assert response.json()["count"] == 3
        assert client.post("/api/v1/predict", json={"text": "   "}).status_code == 422


def test_balanced_opposite_emotions_are_neutral() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict", json={"text": "Estoy feliz y enojado"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] == "neutral"
        assert body["mixed_emotions"] is True
        assert body["probabilities"]["neutral"] > body["probabilities"]["positivo"]
        assert body["probabilities"]["neutral"] > body["probabilities"]["negativo"]


def test_happy_and_anxious_are_balanced_as_neutral() -> None:
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/predict", json={"text": "Me siento feliz y ansioso"}
        ).json()
        assert body["sentiment"] == "neutral"
        assert body["mixed_emotions"] is True
        assert body["probabilities"]["neutral"] >= 0.80
        assert set(body["emotions_detected"]) == {"alegría", "ansiedad"}


def test_intensity_breaks_a_mixed_emotion_tie() -> None:
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/predict",
            json={"text": "Me siento muy feliz y un poco ansioso"},
        ).json()
        assert body["sentiment"] == "positivo"
        assert body["mixed_emotions"] is True


def test_anxiety_without_positive_counterweight_is_negative() -> None:
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/predict", json={"text": "Me siento ansioso y preocupado"}
        ).json()
        assert body["sentiment"] == "negativo"
        assert body["probabilities"]["negativo"] > 0.75


def test_complex_balanced_paragraph_is_neutral() -> None:
    text = (
        "Me gusta el nuevo diseño y algunas funciones son excelentes. "
        "Sin embargo, la aplicación ahora falla, tarda demasiado y eso me enoja. "
        "En general encuentro aspectos buenos y malos en proporciones similares."
    )
    with TestClient(app) as client:
        body = client.post("/api/v1/predict", json={"text": text}).json()
        assert body["sentiment"] == "neutral"
        assert body["mixed_emotions"] is True
        assert body["segments_analyzed"] >= 3


def test_long_paragraph_keeps_dominant_sentiment() -> None:
    positive_text = (
        "La entrega tardó un poco, pero el producto llegó en perfecto estado. "
        "La calidad es excelente, funciona muy bien y el soporte fue amable. "
        "Después de varios días de uso estoy feliz y lo recomiendo totalmente."
    )
    negative_text = (
        "El diseño parece bonito, pero el sistema falla constantemente. "
        "Perdí información importante, la atención fue grosera y nadie solucionó nada. "
        "La experiencia terminó siendo horrible y no lo recomiendo."
    )
    with TestClient(app) as client:
        positive = client.post("/api/v1/predict", json={"text": positive_text}).json()
        negative = client.post("/api/v1/predict", json={"text": negative_text}).json()
        assert positive["sentiment"] == "positivo"
        assert negative["sentiment"] == "negativo"
        assert positive["segments_analyzed"] >= 3
        assert negative["segments_analyzed"] >= 3


def test_sarcastic_negative_context() -> None:
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/predict",
            json={"text": "Qué maravilla, otra vez se borraron todos mis archivos"},
        ).json()
        assert body["sentiment"] == "negativo"
        assert body["sarcasm_detected"] is True


def test_explicitly_undecided_comment_is_neutral() -> None:
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/predict",
            json={
                "text": "Me encantó una parte y odié la otra; tengo sentimientos encontrados"
            },
        ).json()
        assert body["sentiment"] == "neutral"
        assert body["mixed_emotions"] is True
        assert abs(sum(body["probabilities"].values()) - 1) < 0.01
