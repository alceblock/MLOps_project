from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

from model.model_utility import MODEL_PATH

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
    result = sentiment_task(data.text)[0]

    return {
        "label": result["label"],
        "confidence": result["score"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)