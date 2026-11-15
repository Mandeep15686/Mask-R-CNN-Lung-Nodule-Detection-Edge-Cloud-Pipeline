FROM python:3.11-slim

WORKDIR /app

# Runtime libraries for scipy and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend + training package
COPY backend/ /app/backend/
COPY training/ /app/training/

WORKDIR /app/backend

ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:////data/lung_nodule.db
ENV MODEL_PATH=/app/experiments/best_model.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
