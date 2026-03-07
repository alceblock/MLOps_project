from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

from model.model_utility import LATEST_MODEL_PATH

app = FastAPI()

# model_path = "./my_model_versions"
# #model_path = "./my_model_versions/model_v_{}"

sentiment_task = pipeline(
    "sentiment-analysis",
    model=LATEST_MODEL_PATH,
    tokenizer=LATEST_MODEL_PATH
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextInput):
    result = sentiment_task(data.text)[0]

    return {
        "label": result["label"],
        "confidence": result["score"]
    }