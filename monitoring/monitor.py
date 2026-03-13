# ##### model_inference update if reply is ok
# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline
# from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
# from fastapi.responses import Response
# import time
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# app = FastAPI()

# # --- Prometheus metrics ---
# REQUEST_COUNT = Counter("request_count", "Total number of requests")
# REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")
# POSITIVE_COUNT = Counter("positive_count", "Number of positive predictions")
# NEGATIVE_COUNT = Counter("negative_count", "Number of negative predictions")
# NEUTRAL_COUNT = Counter("neutral_count", "Number of neutral predictions")
# TRUE_LABEL_COUNT = Counter("true_label_count", "Number of predictions with true label")
# CORRECT_PREDICTIONS = Counter("correct_predictions", "Number of correct predictions")

# CONFIDENCE_GAUGE = Gauge("confidence_score", "Confidence score of last prediction")

# # --- Sentiment model ---
# sentiment_task = pipeline("sentiment-analysis", model="MODEL_PATH", tokenizer="MODEL_PATH")

# class TextInput(BaseModel):
#     text: str
#     true_label: str | None = None  # optional

# # --- Metrics endpoint ---
# @app.get("/metrics")
# def metrics():
#     return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# @app.post("/predict")
# async def predict(data: TextInput):
#     REQUEST_COUNT.inc()
#     start_time = time.time()

#     result = sentiment_task(data.text)[0]
#     latency = time.time() - start_time
#     REQUEST_LATENCY.observe(latency)

#     label = result["label"].lower()
#     score = result["score"]
#     CONFIDENCE_GAUGE.set(score)

#     # --- Distribuzione delle classi ---
#     if label == "positive":
#         POSITIVE_COUNT.inc()
#     elif label == "negative":
#         NEGATIVE_COUNT.inc()
#     else:
#         NEUTRAL_COUNT.inc()

#     # --- Metriche di accuratezza se c'è true_label ---
#     accuracy = None
#     precision = None
#     recall = None
#     f1 = None

#     if data.true_label:
#         TRUE_LABEL_COUNT.inc()
#         y_true = [data.true_label.lower()]
#         y_pred = [label]
#         if label == y_true[0]:
#             CORRECT_PREDICTIONS.inc()

#         # calcolo metriche sklearn
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#         recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#         f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

#     return {
#         "label": result["label"],
#         "confidence": score,
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "latency": latency
#     }