# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline

## - s
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
## - e

from model_app.model_utility import MODEL_PATH

## - s
from model_app.model_utility import MODEL_PATH
from monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, POSITIVE_COUNT, NEGATIVE_COUNT, NEUTRAL_COUNT
## - e

app = FastAPI()

sentiment_task = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextInput):
    ## - s
    REQUEST_COUNT.inc()  # increment total requests
    start = time.time()
    ## - e

    result = sentiment_task(data.text)[0]

    ## - s
    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)  # latency
    # increment counters per label
    label = result["label"].lower()
    if label == "positive":
        POSITIVE_COUNT.inc()
    elif label == "negative":
        NEGATIVE_COUNT.inc()
    else:
        NEUTRAL_COUNT.inc()
    ## - e

    return {
        "label": result["label"],
        "confidence": result["score"]
    }

## - s
# Prometheus Endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
## - e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)