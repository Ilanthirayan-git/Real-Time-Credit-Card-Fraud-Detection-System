import os
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import shap

from preprocessing import load_or_generate_data, preprocess_and_split, apply_smote

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate():
    # 1. Data Prep
    # The user request specifically asked for data/transactions.csv
    data_path = os.path.join("data", "transactions.csv")
    df = load_or_generate_data(data_path)
    X, y, scaler = preprocess_and_split(df)
    
    # Save the scaler for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. SMOTE on training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 3. Model Definition (Random Forest + XGBoost)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=100, max_depth=6)
    }
    
    best_model_name = None
    best_model = None
    best_f1 = -1
    
    # 4. Train & Eval
    print("--- Training Models ---")
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        logging.info(f"{name} metrics -> Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"{name}:\n{classification_report(y_test, y_pred)}")
        
        # Select best model based on F1 Score to balance precision and recall
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    logging.info(f"Best Model Selected: {best_model_name} with F1: {best_f1:.4f}")
    
    # 5. Save Best Model and feature names
    model_path = 'models/best_model.joblib'
    joblib.dump(best_model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    joblib.dump(X_train.columns.tolist(), 'models/feature_names.joblib')
    
    # Create and save a SHAP explainer
    logging.info("Fitting SHAP explainer on background data...")
    # Use a small background dataset for explainer to speed up inference
    background = shap.sample(X_train, 100)
    
    try:
        if best_model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.KernelExplainer(best_model.predict_proba, background)
        joblib.dump(explainer, 'models/explainer.joblib')
        logging.info("SHAP explainer saved to models/explainer.joblib")
    except Exception as e:
        logging.error(f"Failed to generate explainer: {str(e)}")

if __name__ == "__main__":
    # execute only if run as a script
    train_and_evaluate()
