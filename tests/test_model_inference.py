from fastapi.testclient import TestClient


from model.model_inference import app
client = TestClient(app)

def test_root_endpoint():
    data = {
        "text": "I love this project. It's awsome!"
    }

    response = client.post("/predict", json=data)
    body = response.json()
    # response
    assert response.status_code == 200, response.json
    # label
    assert "label" in body
    assert isinstance(body["label"], str)
    assert body["label"] in ["positive", "neutral", "negative"]
    # confidence
    assert "confidence" in body
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
