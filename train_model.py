import json

from app.model import MODEL_PATH, train_and_save


if __name__ == "__main__":
    metrics = train_and_save()
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

