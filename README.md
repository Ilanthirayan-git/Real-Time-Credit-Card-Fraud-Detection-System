# 🛡️ Real-Time Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-orange)

A complete, production-ready, end-to-end Machine Learning pipeline that detects fraudulent credit card transactions in real-time. This project handles highly imbalanced datasets, offers a full inference backend via FastAPI, and provides a polished interactive frontend via Streamlit for both individual and batch predictions.

---

## 🎯 Problem Statement
Credit card fraud is a critical issue causing billions of dollars in losses annually. The core challenge in detecting it is that fraudulent transactions are extremely rare compared to legitimate ones (often < 1%). This project designs a highly sensitive, explainable machine learning system capable of identifying these rare anomalies in real-time, utilizing synthetic data generation to mimic the European Cardholder Kaggle dataset.

---

## ✨ Features
* **Automated Data Synthesis**: Bootstraps realistic, imbalanced credit card data natively without needing Kaggle credentials.
* **Imbalance Handling**: Combats class imbalance utilizing SMOTE (Synthetic Minority Over-sampling Technique).
* **Model Benchmarking**: Automatically compares multiple tree-based algorithms (Random Forest, XGBoost) to find the best performing F1-Score predictor.
* **Explainable AI (XAI)**: Native SHAP integration clearly visualizes *why* a transaction was flagged as fraudulent based on its features.
* **Real-Time API (Backend)**: Fast, asynchronous REST API using `FastAPI` with rigorous `Pydantic` input validation.
* **Interactive Dashboard (Frontend)**: User-friendly `Streamlit` app allowing manual feature tampering and batch processing via CSV uploads.
* **Production Ready**: Fully dockerized environment with unit tests.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn
* **Explainability:** SHAP, Matplotlib
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Requests

---

## 🏛️ Architecture Explanation
1. **Data Pipeline**: The `preprocessing` module scales the raw dollar `Amount` and transaction `Time` while generating synthetic data.
2. **Training Orchestration**: The `train_model` script actively rebalances datasets, trains tree models, logs evaluation metrics (Recall, Precision, F1), and saves artifacts (model & scaler) to disk via `.joblib`.
3. **Inference Flow**: Incoming API requests are parsed into a schema. Features are transformed by the cached scaler, passed to the best model, and then routed to the SHAP explainer to calculate feature importance weights.
4. **Presentation Layer**: Streamlit consumes the FastAPI `/predict` JSON responses and visualizes predictions via alert components and Matplotlib bar charts.

---

## 📂 Folder Structure
```text
fraud-detection/
│
├── api/                   # FastAPI Backend
│   └── app.py             # Server code and routing
│
├── dashboard/             # Streamlit Frontend
│   └── app.py             # Dashboard UI logic
│
├── data/                  # Datasets & examples
│   ├── sample_request.json# Sample JSON for API testing
│   
├── models/                # Saved ML artifacts (gitignored)
│   ├── best_model.joblib
│
├── src/                   # Machine learning core
│   ├── preprocessing.py   # Transformation & data ops
│   ├── train_model.py     # ML orchestration script
│   └── predict.py         # Inference inference functions
│
├── tests/                 # Built-in unit tests
│   └── test_api.py        # Validates endpoints with Mock Inference
