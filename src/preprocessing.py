import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_data(n_samples=50000, fraud_ratio=0.005, random_seed=42):
    """
    Generates a synthetic dataset similar to the Kaggle Credit Card Fraud dataset.
    Has Time, V1-V28 (PCA-like features), Amount, and Class.
    """
    logging.info(f"Generating synthetic data: {n_samples} samples, {fraud_ratio*100}% fraud...")
    np.random.seed(random_seed)
    
    # Generate background (Class 0)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Time (uniform distribution mimicking 2 days in seconds)
    time_normal = np.random.uniform(0, 172800, n_normal)
    time_fraud = np.random.uniform(0, 172800, n_fraud)
    
    # V1-V28 (Standard normal distribution)
    # Fraud cases have shifted means for some features to make them detectable but overlapping
    v_normal = np.random.randn(n_normal, 28)
    v_fraud = np.random.randn(n_fraud, 28)
    
    # Shift means for fraud on a few features (e.g., V4, V11, V12, V14, V17 - prominent in real dataset)
    shift_indices = [3, 10, 11, 13, 16] # 0-indexed for V4, V11, V12, V14, V17
    for idx in shift_indices:
        v_fraud[:, idx] += np.random.choice([-3, 3]) # Shift distribution
        
    # Amount (Log-normal distribution)
    amount_normal = np.random.lognormal(mean=3, sigma=1.2, size=n_normal)
    amount_fraud = np.random.lognormal(mean=4, sigma=1.5, size=n_fraud)
    
    # Combine
    time = np.concatenate([time_normal, time_fraud])
    v_features = np.vstack([v_normal, v_fraud])
    amount = np.concatenate([amount_normal, amount_fraud])
    classes = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    data = pd.DataFrame(v_features, columns=[f'V{i+1}' for i in range(28)])
    data.insert(0, 'Time', time)
    data['Amount'] = amount
    data['Class'] = classes.astype(int)
    
    # Shuffle
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return data

def load_or_generate_data(data_path="data/creditcard.csv"):
    if os.path.exists(data_path):
        logging.info(f"Loading dataset from {data_path}")
        return pd.read_csv(data_path)
    else:
        logging.warning("Real dataset not found. Generating synthetic dataset.")
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        return df

def preprocess_and_split(df, target_col='Class'):
    """
    Separates features and target, scales Time and Amount, and returns X and y.
    """
    # Scale Time and Amount simultaneously
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, scaler

def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE to balance the training data.
    """
    logging.info(f"Class distribution before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logging.info(f"Class distribution after SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
    return X_train_res, y_train_res
