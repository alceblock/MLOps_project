from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

model_path = "./my_finetuned_model"

sentiment_task = pipeline(
    "sentiment-analysis",
    model=model_path,
    tokenizer=model_path
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