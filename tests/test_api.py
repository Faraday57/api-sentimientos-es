from fastapi.testclient import TestClient

from app.main import app


def test_health_and_model_info() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}
        info = client.get("/api/v1/model/info")
        assert info.status_code == 200
        assert info.json()["dataset_size"] >= 30


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

