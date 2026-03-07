from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

from model.model_utility import MODEL_PATH

app = FastAPI()

# model_path = "./my_model_versions"
# #model_path = "./my_model_versions/model_v_{}"

sentiment_task = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

class TextInput(BaseModel):
    text: str

#####
from prometheus_fastapi_instrumentator import Instrumentator
import time

# Inizializzazione prometheus instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
#####

@app.post("/predict")
async def predict(data: TextInput):
    #####
    start_time = time.time()
    #####
    result = sentiment_task(data.text)[0]

    #####
    latency = time.time() - start_time

    # Custom metrics, es. latency
    from prometheus_client import Summary
    REQUEST_LATENCY = Summary('model_inference_latency_seconds', 'Latency per inference')
    REQUEST_LATENCY.observe(latency)
    #####

    return {
        "label": result["label"],
        "confidence": result["score"]
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)