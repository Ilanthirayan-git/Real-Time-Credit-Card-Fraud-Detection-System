FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Just need the directory structure
COPY . /app

# Ensure correct python path for imports
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8000
EXPOSE 8501

# The default command runs the FastAPI server
# You can override this to run Streamlit
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
