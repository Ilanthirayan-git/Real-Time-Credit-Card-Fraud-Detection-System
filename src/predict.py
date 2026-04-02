import os
import joblib
import pandas as pd
import numpy as np

# Load trained models and dependencies
MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"
EXPLAINER_PATH = "models/explainer.joblib"

# Global lazy loading variables
model = None
scaler = None
feature_names = None
explainer = None

def load_artifacts():
    global model, scaler, feature_names, explainer
    # Only load if not already loaded
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing {MODEL_PATH}. Have you trained the model?")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        try:
            if os.path.exists(EXPLAINER_PATH):
                explainer = joblib.load(EXPLAINER_PATH)
        except Exception:
            explainer = None # Soft fail on explainer

def predict_transaction(tx_data_dict):
    """
    Predicts if a single transaction is fraudulent.
    tx_data_dict should be a dictionary with keys: Time, Amount, V1-V28.
    """
    load_artifacts()
    
    # Needs to match exactly the column order during training
    # For preprocessing: we only scaled Time and Amount
    
    df = pd.DataFrame([tx_data_dict])
    
    # Apply standard scaling (assuming Time and Amount were scaled using the same specific scaler)
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    
    # Ensure correct column order
    X = df[feature_names]
    
    # Prediction
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return {
        "prediction": int(pred),
        "probability": float(prob)
    }

def get_shap_values(tx_data_dict):
    """
    Returns SHAP values for the given transaction.
    """
    load_artifacts()
    
    if explainer is None:
        return None
        
    df = pd.DataFrame([tx_data_dict])
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    X = df[feature_names]
    
    shap_v = explainer.shap_values(X)
    return shap_v.tolist()[0] if isinstance(shap_v, np.ndarray) else shap_v[0].tolist()

if __name__ == "__main__":
    # Test predict
    test_tx = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        test_tx[f"V{i}"] = 0.0
    print("Test inference:", predict_transaction(test_tx))
