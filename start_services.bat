@echo off
echo Starting FastAPI Backend...
set PYTHONPATH=src
start cmd /k ".\venv\Scripts\activate && uvicorn api.app:app --reload"

echo Starting Streamlit Dashboard...
set PYTHONPATH=src
start cmd /k ".\venv\Scripts\activate && streamlit run dashboard/app.py"
