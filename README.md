# API de análisis de sentimientos en español

Proyecto académico de Inteligencia Artificial y Machine Learning. Entrena un modelo supervisado para clasificar comentarios de Facebook u otras plataformas como **positivos**, **neutrales** o **negativos**, y lo expone mediante una API REST con FastAPI.

## ¿Cómo funciona?

El entrenamiento transforma el texto con dos representaciones TF-IDF: n-gramas de palabras (capturan expresiones) y n-gramas de caracteres (toleran variantes, errores y palabras desconocidas). Estas representaciones se combinan con señales lingüísticas de polaridad, negación, intensidad y contraste. Una regresión logística multiclase produce la primera estimación.

Encima de esa estimación se ejecuta una capa de **inferencia difusa contextual**. Cada emoción aporta evidencia gradual positiva o negativa; `muy` aumenta su peso, `poco` lo reduce y una negación invierte su valencia. Las familias morfológicas permiten reconocer variantes como `ansioso`, `ansiosa` y `ansiedad`. Si ambos polos tienen intensidad semejante, aumenta la pertenencia neutral; si uno domina por cantidad o intensidad, conserva su polaridad. Por ejemplo, `feliz y ansioso` es neutral, mientras que `muy feliz y un poco ansioso` es positivo.

La versión 5 se entrena con **1.500 ejemplos balanceados** (500 por clase): 180 comentarios curados y 1.320 variaciones sintéticas reproducibles. Incluye comentarios largos, emociones contradictorias, ansiedad, opiniones con ventajas y desventajas, negaciones, intensidad, expresiones neutrales y patrones frecuentes de sarcasmo. Para analizar párrafos, el servicio separa oraciones y cláusulas introducidas por conectores como `pero`, `aunque` y `sin embargo`; la conclusión posterior a un contraste recibe algo más de peso.

> Este es un modelo académico entrenado con un conjunto pequeño. No debe emplearse para tomar decisiones sobre personas ni como única fuente para moderación real.

## Ejecutar localmente

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python train_model.py
uvicorn app.main:app --reload
```

Abrir `http://127.0.0.1:8000` para usar la interfaz o `http://127.0.0.1:8000/docs` para Swagger.

## Consumir la API

Un comentario:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Me encantó el servicio, muchas gracias"}'
```

Respuesta:

```json
{
  "text": "Me encantó el servicio, muchas gracias",
  "sentiment": "positivo",
  "confidence": 0.7812,
  "mixed_emotions": false,
  "sarcasm_detected": false,
  "segments_analyzed": 1,
  "emotions_detected": ["satisfacción"],
  "probabilities": {"positivo": 0.7812, "neutral": 0.128, "negativo": 0.0908}
}
```

Rutas disponibles:

| Método | Ruta | Uso |
|---|---|---|
| GET | `/` | Interfaz web |
| GET | `/docs` | Documentación Swagger |
| GET | `/health` | Estado de la API |
| GET | `/api/v1/model/info` | Métricas y datos del modelo |
| POST | `/api/v1/predict` | Clasificar un comentario |
| POST | `/api/v1/predict/batch` | Clasificar hasta 100 comentarios |

## Pruebas

```powershell
pytest -q
```

## Desplegar gratis en Render

1. Subir esta carpeta a un repositorio de GitHub.
2. En [Render](https://render.com), elegir **New > Blueprint** y conectar el repositorio.
3. Render detectará `render.yaml`; confirmar la creación del servicio.
4. Al terminar, copiar la URL pública similar a `https://api-sentimientos-es.onrender.com`.

También se incluye un `Dockerfile` para cualquier plataforma compatible con contenedores. En planes gratuitos, el primer acceso después de un periodo de inactividad puede tardar mientras el servicio inicia.

## Reentrenamiento

Agregar ejemplos a `data/sentiment_es.jsonl` respetando el formato JSON Lines y ejecutar:

```powershell
python train_model.py
```

El script amplía el corpus de forma determinista hasta 500 ejemplos por clase, entrena el artefacto y evalúa el sistema híbrido con `data/challenge_es.jsonl`. Ese conjunto contiene 45 frases y párrafos complejos escritos por separado y nunca se usa para entrenar. La evaluación actual obtuvo aproximadamente **100 % de exactitud y F1 macro** en ese reto local. Es una prueba académica pequeña y dirigida; no demuestra rendimiento perfecto en comentarios reales ni sustituye una evaluación humana amplia.

## Referencias técnicas

- L. A. Zadeh, [Fuzzy Sets](https://doi.org/10.1016/S0019-9958(65)90241-X): pertenencia gradual entre 0 y 1.
- Hutto y Gilbert, [VADER](https://doi.org/10.1609/icwsm.v8i1.14550): combinación de léxico con reglas contextuales para texto de redes sociales.
- Documentación oficial de [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) y [regresión logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) de scikit-learn.
