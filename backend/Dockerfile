FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY artifacts /app/artifacts

ENV PYTHONPATH=/app/backend \
    MODEL_PATH=/app/artifacts/fashion_cnn.pt

EXPOSE 8765

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]

