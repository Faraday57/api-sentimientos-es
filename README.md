# API de análisis de sentimientos en español

Proyecto académico de Inteligencia Artificial y Machine Learning. Entrena un modelo supervisado para clasificar comentarios de Facebook u otras plataformas como **positivos**, **neutrales** o **negativos**, y lo expone mediante una API REST con FastAPI.

## ¿Cómo funciona?

El entrenamiento transforma el texto con dos representaciones TF-IDF: n-gramas de palabras (capturan expresiones) y n-gramas de caracteres (toleran variantes, errores y palabras desconocidas). Estas representaciones se combinan con señales lingüísticas de polaridad, negación, intensidad y contraste. Una regresión logística multiclase produce la etiqueta y las probabilidades.

La versión actual contiene **180 ejemplos balanceados** en español, incluyendo comentarios largos, emociones contradictorias, opiniones con ventajas y desventajas, negaciones, expresiones neutrales y algunos patrones frecuentes de sarcasmo. Para analizar párrafos, el servicio separa oraciones y cláusulas introducidas por conectores como `pero`, `aunque` y `sin embargo`; después combina el resultado global con los resultados de cada segmento. Si la evidencia positiva y negativa tiene intensidad semejante, el tono global se interpreta como mixto o neutral.

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

El script divide los datos de manera estratificada, calcula exactitud y F1 macro en el conjunto de prueba, y después entrena el artefacto final con todos los ejemplos. La evaluación actual utiliza 144 ejemplos para entrenamiento y 36 para prueba; obtuvo aproximadamente **86.1 % de exactitud** y **86.5 % de F1 macro**. Son métricas académicas sobre un conjunto pequeño y no garantizan el mismo rendimiento en todos los comentarios reales.
