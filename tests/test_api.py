import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure api and src are in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app
import predict

client = TestClient(app)

def mock_predict_transaction(tx_data_dict):
    # Mocking the inference function so tests run without trained .joblib models
    return {
        "prediction": 1,
        "probability": 0.99
    }

def mock_get_shap_values(tx_data_dict):
    # Mock SHAP values array
    return [0.1] * 30

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(monkeypatch):
    # Apply monkeypatching to bypass actual ML model inference
    monkeypatch.setattr(predict, "predict_transaction", mock_predict_transaction)
    monkeypatch.setattr(predict, "get_shap_values", mock_get_shap_values)

    payload = {
        "Time": 0.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62
    }

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == 0.99
    assert "execution_time_sec" in data
    assert len(data["shap_values"]) == 30

def test_predict_invalid_payload():
    # Missing 'Amount' field
    payload = {
        "Time": 0.0,
        "V1": -1.359807 # incomplete
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # FastAPI validation error
