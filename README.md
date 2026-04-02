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
* **Testing:** Pytest, HTTPX
* **Deployment:** Docker

---

## 🏛️ Architecture Explanation
1. **Data Pipeline**: The `preprocessing` module scales the raw dollar `Amount` and transaction `Time` while generating synthetic data.
2. **Training Orchestration**: The `train_model` script actively rebalances datasets, trains tree models, logs evaluation metrics (Recall, Precision, F1), and saves artifacts (model & scaler) to disk via `.joblib`.
3. **Inference Flow**: Incoming API requests are parsed into a schema. Features are transformed by the cached scaler, passed to the best model, and then routed to the SHAP explainer to calculate feature importance weights.
4. **Presentation Layer**: Streamlit consumes the FastAPI `/predict` JSON responses and visualizes predictions via alert components and Matplotlib bar charts.

---

## 📸 Screenshots

| Manual Inference | Batch Evaluation Dashboard |
| :---: | :---: |
| ![manual](https://via.placeholder.com/400x250.png?text=Manual+Transaction+Input) | ![batch](https://via.placeholder.com/400x250.png?text=Batch+CSV+Upload+Dashboard) |
| *Form inputs validating user metrics* | *Visualizing Fraud Distribution over time* |

*(Note: Replace with actual screenshots of your dashboard before portfolio publication)*

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
│   └── transactions.csv   # Auto-generated dataset (gitignored)
│   
├── models/                # Saved ML artifacts (gitignored)
│   ├── best_model.joblib
│   ├── explainer.joblib
│   └── scaler.joblib
│
├── src/                   # Machine learning core
│   ├── preprocessing.py   # Transformation & data ops
│   ├── train_model.py     # ML orchestration script
│   └── predict.py         # Inference inference functions
│
├── tests/                 # Built-in unit tests
│   └── test_api.py        # Validates endpoints with Mock Inference
│
├── .gitignore             # Git exclusions
├── Dockerfile             # Container configuration
├── requirements.txt       # Software dependencies
└── README.md              # Project documentation
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/Real-Time-Fraud-Detection.git
cd Real-Time-Fraud-Detection
```

### 2. Prepare Virtual Environment
Create and activate an isolated Python environment:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Data and Train the Model
Synthesize the data and export model artifacts:
```bash
python src/train_model.py
```
> *Ensure `.joblib` files appear in the `models/` directory upon completion.*

### 5. Start the Application
You will need two separate terminal windows for the frontend and backend. Both must have the `venv` active.

**Terminal 1 (Backend API):**
```bash
# Set path to allow relative imports
# Windows: set PYTHONPATH=src | Unix: export PYTHONPATH=src
set PYTHONPATH=src
uvicorn api.app:app --reload
```
The API is now running at `http://127.0.0.1:8000`. You can access auto-generated API docs at `http://127.0.0.1:8000/docs`.

**Terminal 2 (Frontend Dashboard):**
```bash
set PYTHONPATH=src
streamlit run dashboard/app.py
```

---

## 🔌 API Usage Example

The backend serves a raw inference predicting engine. Test it using Python `requests` or `curl`.

### cURL Request
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/sample_request.json
```

### Example JSON Payload 
*(See `data/sample_request.json` for full format)*
```json
{
  "Time": 0.0,
  "V1": -1.359807,
  "Amount": 149.62
  ...
}
```

### Example Response
```json
{
  "prediction": 1,
  "fraud_probability": 0.985,
  "is_fraud": true,
  "execution_time_sec": 0.041
}
```

---

## 🔮 Future Improvements
- **Kafka Integration**: Implement Apache Kafka to consume continuous streams of transaction data instead of HTTP POST batches.
- **Model Versioning**: Implement MLflow logging to track hyperparameter tuning and model registry gracefully.
- **Advanced Unsupervised Learning**: Train deep autoencoders as native backends for anomalous behavior detection alongside tree ensembles.
- **Cloud Database Integration**: Write verified transactions directly to a Postgres or MongoDB cluster for historical querying on the dashboard.
