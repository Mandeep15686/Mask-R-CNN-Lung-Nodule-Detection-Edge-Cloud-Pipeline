# 🫁 Mask R-CNN Lung Nodule Detection — Edge–Cloud Pipeline

> Full-stack research system for LUNA16 lung nodule detection and segmentation.  
> **DIP Track**: Composite loss weight optimisation (BCE + Dice + Boundary + Focal + HD95)  
> **CN Track**: Edge–Cloud split-computing pipeline with feature compression and federated learning

---

## 📁 Project Structure

```
lung-nodule/
├── training/           # DIP: Model, losses, dataset, trainer, experiments
│   ├── losses.py       # BCE, Dice, Boundary, Focal, Tversky, HD95 losses
│   ├── dataset.py      # LUNA16 CT loader + 2.5D patch extraction
│   ├── model.py        # Mask R-CNN with custom mask loss injection
│   ├── metrics.py      # Dice, IoU, HD95, Boundary F1, FROC
│   ├── trainer.py      # Full training loop + checkpointing
│   └── experiment.py   # Phase 1/2/3 experiment runner
│
├── backend/            # FastAPI cloud server
│   └── app/
│       ├── main.py     # App factory + lifespan
│       ├── database.py # SQLAlchemy models (Experiment, Run, Inference, Edge)
│       ├── schemas.py  # Pydantic v2 schemas
│       ├── routes/     # inference, experiments, results, edge
│       └── ml/         # InferenceEngine
│
├── edge/               # CN: Edge node service
│   ├── edge_node.py    # Async edge controller + HTTP server
│   ├── preprocessor.py # CT HU windowing
│   ├── feature_extractor.py  # Early ResNet layers (split point)
│   ├── compressor.py   # INT8 / learned feature compression
│   └── network_monitor.py    # Bandwidth + latency probing
│
├── frontend/           # React dashboard
│   └── src/
│       ├── pages/      # Dashboard, Experiments, Leaderboard, Inference, EdgeNetwork
│       └── services/   # Axios API layer + WebSocket
│
├── scripts/
│   ├── run_experiments.py    # CLI for Phase 1/2/3
│   └── simulate_edge.py      # Edge-cloud pipeline simulator
│
├── docker-compose.yml
└── .env.example
```

---

## ⚡ Quick Start (Docker — Recommended)

### 1. Prerequisites
```bash
# Docker + Docker Compose
docker --version        # >= 24.0
docker compose version  # >= 2.0

# Optional: NVIDIA GPU support
nvidia-smi
```

### 2. Clone and configure
```bash
git clone https://github.com/Mandeep15686/Mask-R-CNN-Lung-Nodule-Detection-Edge-Cloud-Pipeline.git lung-nodule
cd lung-nodule
cp .env.example .env
```

Set `EDGE_API_TOKEN` in `.env` to a long random secret before exposing the backend.
The feature-upload endpoint rejects requests if the token is missing or invalid.
Set `APP_ENV=production` and provide a trained checkpoint at `MODEL_PATH` for a
deployment; production startup fails if the checkpoint is missing. Development
keeps the untrained-model fallback enabled.

### 3. Launch all services
```bash
docker compose up --build
```

| Service   | URL                    | Description          |
|-----------|------------------------|----------------------|
| Frontend  | http://localhost:3000  | React dashboard      |
| Backend   | http://localhost:8000  | FastAPI + Swagger UI |
| Edge Node | http://localhost:9090  | Edge inference HTTP  |
| MLflow    | http://localhost:5000  | (optional) Tracking  |

```bash
# With MLflow:
docker compose --profile mlflow up --build
```

---

## 🔧 Manual Setup (Development)

### Python environment (backend + training)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install training deps
pip install -r training/requirements.txt

# Install backend deps
pip install -r backend/requirements.txt
```

### Node.js (frontend)
```bash
cd frontend
npm install
npm run dev    # Vite dev server at http://localhost:3000
```

### Run the backend
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the edge node
```bash
cd edge
pip install -r requirements.txt
python edge_node.py
```

---

## 📦 LUNA16 Dataset Setup

```bash
mkdir -p data/luna16
# Download from https://luna16.grand-challenge.org/Download/
# Place subsets:
#   data/luna16/subset0/ ... subset9/   (.mhd + .raw files)
#   data/luna16/annotations.csv
#   data/luna16/candidates.csv
```

---

## 🔬 DIP: Running Loss-Weight Experiments

### Phase 1 — Single-loss ablations (6 experiments)
```bash
python scripts/run_experiments.py --phase 1 \
  --data ./data/luna16 --out ./experiments --epochs 30
```

### Phase 2 — Grid search (α, β, γ)
```bash
python scripts/run_experiments.py --phase 2 \
  --data ./data/luna16 --out ./experiments \
  --epochs 30 --max-configs 40
```

### Phase 3 — Bayesian optimisation (Optuna)
```bash
python scripts/run_experiments.py --phase 3 \
  --data ./data/luna16 --out ./experiments --n-trials 100
```

### Via the UI
1. Open **http://localhost:3000/experiments**
2. Click **New Experiment**, set loss weights α/β/γ/δ/ε
3. Click **Start** — live learning curve streams via WebSocket
4. View **Leaderboard** → Pareto frontier scatter plot

### Programmatic API
```python
from training.losses import CompositeMaskLoss

# Best hypothesis: 0.3·BCE + 0.5·Dice + 0.5·Boundary
loss_fn = CompositeMaskLoss(alpha=0.3, beta=0.5, gamma=0.5)

pred   = model_output  # (N, H, W) logits
target = gt_masks      # (N, H, W) binary
loss_dict = loss_fn(pred, target)
# → {"bce": tensor, "dice": tensor, "boundary": tensor, "total": tensor}
loss_dict["total"].backward()
```

---

## 🌐 CN: Edge–Cloud Pipeline

### Test the pipeline locally

The current edge extractor sends one intermediate ResNet tensor, while the
cloud Mask R-CNN continuation requires all five FPN feature levels. The API
rejects this incompatible payload with HTTP 501 instead of returning a
misleading prediction. Edge requests therefore fall back to the local detector.
Use direct cloud inference below until the edge/cloud feature contract is updated.

```bash
# Simulate with a test image
python scripts/simulate_edge.py \
  --image data/sample_patch.png \
  --split layer2 \
  --compress int8

# With real cloud endpoint:
python scripts/simulate_edge.py \
  --image data/sample_patch.png \
  --split layer2 \
  --compress int8 \
  --cloud http://localhost:8000
```

### Submit a scan to the edge node
```bash
curl -X POST "http://localhost:9090/scan?scan_uid=test_001" \
  --data-binary @data/sample_patch.png \
  -H "Content-Type: image/png"
```

### Direct cloud inference (bypass edge)
```bash
curl -X POST "http://localhost:8000/api/inference/upload" \
  -F "scan_uid=test_001" \
  -F "source=cloud" \
  -F "file=@data/sample_patch.png"
```

### Network analytics (CN research)
```
GET /api/edge/split-latency-analysis  → Split-point latency comparison
GET /api/edge/network-stats           → Compression method statistics
GET /api/edge/network-log             → Per-request latency/bandwidth log
GET /api/results/pareto               → Pareto frontier (Dice vs HD95)
```

### Configure edge split point
```bash
# Edit .env or docker-compose.yml:
SPLIT_LAYER=layer1    # Transmits 256ch stride-4 features (~16KB)
SPLIT_LAYER=layer2    # Transmits 512ch stride-8 features (~32KB) ← default
SPLIT_LAYER=layer3    # Transmits 1024ch stride-16 features (~64KB)
COMPRESSION=none      # Raw float32 features
COMPRESSION=int8      # INT8 quantisation (~4× compression)
COMPRESSION=learned   # Requires a trained checkpoint via LEARNED_COMPRESSOR_CHECKPOINT
```

---

## 🧪 Testing

### Backend unit tests
```bash
pip install pytest pytest-asyncio httpx
pytest backend/tests/ -v
```

### Quick API smoke test
```bash
# Health check
curl http://localhost:8000/health

# Create an experiment
curl -X POST http://localhost:8000/api/experiments/ \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "test-dice-boundary",
    "phase": 1,
    "loss_weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.5, "delta": 0, "epsilon": 0},
    "epochs": 5, "batch_size": 2, "fold": 0, "lr": 0.005,
    "data_root": "./data/luna16"
  }'

# List experiments
curl http://localhost:8000/api/experiments/

# Leaderboard
curl http://localhost:8000/api/results/leaderboard
```

### Full API docs
Open http://localhost:8000/docs — interactive Swagger UI.

---

## 📊 Key API Endpoints

| Method | Endpoint                               | Description                      |
|--------|----------------------------------------|----------------------------------|
| POST   | `/api/experiments/`                    | Create experiment                |
| POST   | `/api/experiments/{id}/start`          | Start training (background)      |
| GET    | `/api/experiments/{id}/runs`           | Epoch-by-epoch metrics           |
| WS     | `/api/experiments/{id}/ws`             | Live training progress           |
| POST   | `/api/inference/upload`                | Infer from file upload           |
| GET    | `/api/results/leaderboard`             | All experiments ranked by Dice   |
| GET    | `/api/results/pareto`                  | Pareto frontier (Dice vs HD95)   |
| GET    | `/api/edge/split-latency-analysis`     | CN1: split point optimisation    |
| GET    | `/api/edge/network-stats`              | CN2: compression comparison      |
| POST   | `/api/edge/infer-features`             | Receive edge features, run cloud |

---

## 📐 Loss Function Quick Reference

```
L_mask = α·L_BCE + β·L_Dice + γ·L_Boundary + δ·L_Focal + ε·L_HD95

Recommended configurations:
  Baseline:          α=1.0  β=0   γ=0   (pure BCE — training stable)
  Overlap-focused:   α=0    β=1.0 γ=0   (pure Dice — good for imbalance)
  Balanced (default):α=0.3  β=0.5 γ=0.5 (hypothesised best)
  Boundary-dominant: α=0.3  β=0.3 γ=0.7 (sharp edges, may lower Dice)
  All losses:        α=0.3  β=0.4 γ=0.3 δ=0.1 ε=0.1
```

---

## 🚀 Production Deployment

```bash
# Build production frontend
cd frontend && npm run build

# Run with PostgreSQL
DATABASE_URL=postgresql://user:pass@db:5432/lung_nodule docker compose up

# GPU backend
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

## 📚 References

- **Mask R-CNN**: He et al. (2017) — https://arxiv.org/abs/1703.06870
- **Boundary Loss**: Kervadec et al. (2019) — https://arxiv.org/abs/1812.07032
- **LUNA16**: Setio et al. (2017) — https://doi.org/10.1016/j.media.2017.06.011
- **Hausdorff DT Loss**: Karimi & Salcudean (2019) — https://arxiv.org/abs/1904.10030
- **FedAvg**: McMahan et al. (2017) — https://arxiv.org/abs/1602.05629
- **Split Computing**: Matsubara et al. (2022) — https://arxiv.org/abs/2205.09940
