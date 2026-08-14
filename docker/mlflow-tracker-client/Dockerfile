FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt

COPY mlflow_stock_tracker.py .