# Lung Nodule Detection: Mask R-CNN Edge–Cloud Research Pipeline

A research codebase for 2.5D lung-nodule detection and mask segmentation on LUNA16, with a FastAPI service, React research dashboard, and an edge-node prototype. The project has two research tracks:

- **DIP (Deep Learning):** train and compare Mask R-CNN mask-loss configurations using validation metrics.
- **CN (Computer Networks):** record edge-node and network measurements, and investigate split inference, compression, federated learning, and privacy mechanisms.

This repository is a research prototype. It is **not a clinical product** and its predictions are not medical diagnoses. Read [Research scope and limitations](#research-scope-and-limitations) before interpreting results.

## Contents

- [Repository map](#repository-map)
- [Requirements](#requirements)
- [Quick start with Docker Compose](#quick-start-with-docker-compose)
- [Local development on Windows](#local-development-on-windows)
- [Local development on Linux or macOS](#local-development-on-linux-or-macos)
- [LUNA16 data and patch cache](#luna16-data-and-patch-cache)
- [Training experiments](#training-experiments)
- [Model, losses, and metrics](#model-losses-and-metrics)
- [Evaluation](#evaluation)
- [Web application](#web-application)
- [HTTP and WebSocket API](#http-and-websocket-api)
- [Edge and cloud prototype](#edge-and-cloud-prototype)
- [Other research scripts](#other-research-scripts)
- [Tests and checks](#tests-and-checks)
- [Configuration reference](#configuration-reference)
- [Data, checkpoints, and Git](#data-checkpoints-and-git)
- [Research scope and limitations](#research-scope-and-limitations)

## Repository map

```text
.
├── backend/
│   ├── run.py                   # Standalone launcher with environment checks
│   ├── alembic/                 # Initial database migration
│   └── app/
│       ├── main.py              # FastAPI app, startup, router registration
│       ├── database.py          # SQLAlchemy entities and DB setup
│       ├── data_config.py       # Data/cache path resolution and cache validation
│       ├── schemas.py           # Pydantic API request/response schemas
│       ├── ml/inference.py      # Strict checkpoint-backed inference
│       ├── json_utils.py        # JSON-safe API/database values
│       ├── utils/preprocessing.py # Strict uploaded image preprocessing
│       ├── routes/dashboard.py # Health, summary, dataset and activity APIs
│       ├── routes/experiments.py # CRUD, training, logs and WebSocket
│       ├── routes/inference.py # Upload, predict and inference history
│       ├── routes/results.py   # Stats, curves, leaderboard and Pareto
│       └── routes/edge.py      # Nodes, heartbeats and network APIs
├── configs/                     # Example Phase 1/2/3 configurations
├── edge/
│   ├── edge_node.py             # Edge HTTP service and cloud communication
│   ├── feature_extractor.py     # ResNet feature extraction at split points
│   ├── compressor.py            # Feature quantization/compression experiments
│   ├── network_monitor.py       # Network measurement helpers
│   └── preprocessor.py          # Input windowing/preprocessing
├── frontend/
│   ├── package.json            # React/Vite dependencies and commands
│   ├── package-lock.json
│   ├── eslint.config.js
│   ├── src/App.jsx              # Navigation and page routes
│   ├── src/pages/               # Dashboard, experiments, leaderboard, etc.
│   ├── src/services/api.js      # HTTP and training WebSocket client
│   ├── vite.config.js           # Dev server and backend proxy
│   └── nginx.conf               # Container web server and API proxy
├── scripts/
│   ├── run_experiments.py       # Phase 1/2/3 training CLI
│   ├── evaluate_model.py        # Checkpoint evaluation CLI
│   ├── cn1_split_analysis.py    # Simulated network-profile split analysis
│   ├── cn5_differential_privacy.py
│   ├── cn5_dp_patch.py           # Experimental patch helper
│   ├── simulate_edge.py         # Edge pipeline simulator
│   ├── test_inference.py        # Inference utility script
│   ├── import_experiment_folders.py
│   ├── plot_results.py
│   └── generate_test_data.py   # Synthetic development fixtures
├── training/
│   ├── __init__.py
│   ├── model.py                 # Torchvision Mask R-CNN, ResNet-50-FPN
│   ├── losses.py                # Composite mask losses
│   ├── dataset.py               # Raw and cached LUNA16 datasets
│   ├── preprocess_cache.py      # Patch-cache build and validation CLI
│   ├── metrics.py               # Segmentation and sampled-patch metrics
│   ├── trainer.py               # Training loop and checkpoints
│   ├── experiment.py            # Phase orchestration and Phase 2 resume
│   ├── federated.py             # Flower-based federated simulation/client
│   └── warmup.py                # Model warmup utility
├── tests/                       # Pytest suite
├── docker-compose.yml
├── docker-compose.gpu.yml
├── Makefile
├── .env.example
├── .gitignore
└── README.md
```

Local data contents, the default `experiments/` output, model weight files, SQLite databases, and `.env` secrets are excluded by Git. Some named Phase 2 output folders are not globally ignored, so check `git status` before staging large run artifacts. Their exact contents depend on each developer’s machine.

## System Architecture

![System Architecture](assets/Architecture.png)

[![Architecture diagram of mandeep15686/mask-r-cnn-lung-nodule-detection-edge-cloud-pipeline](https://gitdiagram.com/mandeep15686/mask-r-cnn-lung-nodule-detection-edge-cloud-pipeline/diagram.png)](https://gitdiagram.com/mandeep15686/mask-r-cnn-lung-nodule-detection-edge-cloud-pipeline?utm_source=readme&utm_medium=picture)

## Requirements

- 64-bit Python 3.11 is the tested project environment. GPU training requires a compatible NVIDIA driver and PyTorch CUDA runtime; CPU is supported but training is slower.
- Node.js 20 or newer and npm for frontend development.
- Docker Engine/Desktop with Docker Compose v2 for the container workflow.
- The LUNA16 scans and annotations for real dataset work. Unit tests use synthetic fixtures and do not download LUNA16.

The pinned Python requirements are split by use:

- `training/requirements.txt`: PyTorch, SimpleITK, SciPy, Optuna, Flower, and training/reporting libraries.
- `backend/requirements.txt`: FastAPI, SQLAlchemy, Alembic, PyTorch, image and API dependencies.
- `edge/requirements.txt`: edge service dependencies.

If the pinned Torch build does not match your platform or CUDA setup, install the appropriate PyTorch 2.4.0/torchvision 0.19.0 pair from the official PyTorch selector, then install the remaining requirements. The training project currently targets Python 3.11 wheels on Windows.

## Quick start with Docker Compose

From a clone of this repository:

```bash
git clone https://github.com/Mandeep15686/Mask-R-CNN-Lung-Nodule-Detection-Edge-Cloud-Pipeline.git
cd Mask-R-CNN-Lung-Nodule-Detection-Edge-Cloud-Pipeline
cp .env.example .env
```

Before exposing the services beyond a trusted development machine, replace the example `EDGE_API_TOKEN` with a private random value, protect the database, and configure CORS and network access for your deployment. Do not commit `.env`.

Put the dataset and a validated patch cache in the host `data/` directory. Compose mounts that directory at `/data` and configures the backend to use `/data/luna16` and `/data/patches_cache_v3_smoke` by default. Put trained experiment folders under `experiments/`; the directory is mounted into the backend container. Configure `MODEL_PATH` to a real checkpoint if you want a default model loaded at startup.

```bash
docker compose up --build
```

| Service | Local address | Purpose |
| --- | --- | --- |
| Dashboard | <http://localhost:3000> | React UI served by Nginx |
| Backend | <http://localhost:8000> | FastAPI service |
| API docs | <http://localhost:8000/docs> | Interactive OpenAPI docs |
| Edge node | <http://localhost:9090> | Edge-node HTTP service |

The backend health check gates edge and frontend startup. To start the optional MLflow service:

```bash
docker compose --profile mlflow up --build
```

MLflow is exposed at <http://localhost:5000>. For a Docker Compose deployment with an NVIDIA GPU and configured NVIDIA Container Toolkit, use:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The production frontend image uses Nginx to serve the built app and proxy `/api` requests to the backend. The Vite dev proxy is only for local development.

## Local development on Windows

Run each service in its own PowerShell terminal from the repository root. Configure paths for the backend process so they point to files visible to that process:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r training/requirements.txt
pip install -r backend/requirements.txt
```

Set the local data paths (adjust the drive/path if needed), then start the API:

```powershell
$env:DATA_ROOT = "D:\lung-nodule\data\luna16"
$env:LUNA16_CACHE_DIR = "D:\lung-nodule\data\patches_cache_v3_smoke"
$env:DATABASE_URL = "sqlite:///D:/lung-nodule/data/lung_nodule.db"
$env:EXPERIMENTS_DIR = "D:\lung-nodule\experiments"
Set-Location backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If you want the default inference model loaded, set `MODEL_PATH` to an existing trained checkpoint **before starting the backend**. Without one, the API can start and show the inference service as unavailable; it does not silently use random/untrained weights for predictions. Experiment-specific inference can select a saved checkpoint from an experiment record.

Start the frontend in another terminal:

```powershell
Set-Location D:\lung-nodule\frontend
npm ci
npm run dev
```

Open <http://localhost:3000>. Vite forwards `/api` requests to `http://localhost:8000`.

Optional edge service, in another terminal:

```powershell
Set-Location D:\lung-nodule\edge
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe edge_node.py
```

The edge service uses `CLOUD_API_URL`, `NODE_ID`, `SPLIT_LAYER`, `COMPRESSION`, `EDGE_API_TOKEN`, and `HEARTBEAT_SEC` environment variables. Set them in the same shell before launching when the defaults do not match your setup.

## Local development on Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r training/requirements.txt
pip install -r backend/requirements.txt

export DATA_ROOT="$PWD/data/luna16"
export LUNA16_CACHE_DIR="$PWD/data/patches_cache_v3_smoke"
export DATABASE_URL="sqlite:///$PWD/data/lung_nodule.db"
export EXPERIMENTS_DIR="$PWD/experiments"

cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

For the edge service, install `edge/requirements.txt` in a Python environment that can import the compatible Torch installation, then run `python edge_node.py` from `edge/`.

## LUNA16 data and patch cache

The raw data root is expected to contain LUNA16 subset directories and the annotation tables, for example:

```text
data/luna16/
├── annotations.csv
├── candidates.csv
├── subset0/
│   ├── <LUNA scan UID>.mhd
│   └── <LUNA scan UID>.raw
└── subset1/
    └── ...
```

Obtain LUNA16 from the [official challenge download page](https://luna16.grand-challenge.org/Download/) and follow its terms. The project’s default local split uses an available validation fold and other available subsets for training; for two local subsets, the intended development split is training on `subset1` and validation on `subset0`. It is **not** an independent test split.

Training can read raw scans, but the web experiment workflow defaults to the validated patch cache. Cache manifests record patch labels, scan UID, subset, patch dimensions, spacing, and NPZ file references. The backend validates the cache before accepting and starting a cached experiment. It does not fall back to raw scans when a requested cached split is missing or invalid.

Build a cache only when required. This can take substantial time and disk space:

```powershell
# From repository root; options retain the script's underscore spelling.
python -m training.preprocess_cache `
  --data_root .\data\luna16 `
  --out_dir .\data\patches_cache_v3 `
  --patch_size 64 `
  --workers 4 `
  --target-spacing-mm 1 1 1
```

The builder derives subset metadata from each scan path, validates the completed manifest and referenced NPZ files, checks expected positive/negative counts, and publishes the completed cache only after processing succeeds. An existing complete cache is retained. To deliberately rebuild an existing or partial cache, pass `--rebuild`; the prior directory is moved to a timestamped backup. Do not point `--rebuild` at a cache you need to preserve.

Inspect cache, split, and model readiness in the **Dataset & Models** page or with:

```text
GET http://localhost:8000/api/dashboard/dataset
GET http://localhost:8000/api/info
```

The cache endpoint reports the runtime-resolved paths, validation result, patch counts, dimensions, available subsets, spacing, and split summary where available. These values come from the active backend’s local files, not from constants in this README.

## Training experiments

`scripts/run_experiments.py` supports three phases. Commands below assume a validated cache exists and matches the selected subsets. The `--cache-dir` value must be visible to the training process.

### Phase 1: single-loss ablations

Runs BCE, Dice, Boundary, Focal, and distance-transform Hausdorff surrogate configurations:

```powershell
python scripts/run_experiments.py `
  --phase 1 `
  --data .\data\luna16 `
  --cache-dir .\data\patches_cache_v3 `
  --train-subsets 1 `
  --validation-subset 0 `
  --out .\experiments `
  --epochs 30 `
  --batch-size 4
```

### Phase 2: loss-weight grid

The full grid tests non-zero combinations of alpha, beta, and gamma over five candidate values per weight. `--max-configs` selects a deterministic random sample using seed 42. `--resume` reuses a result only when its summary, expected configuration ID, completed epoch history, and latest checkpoint are present and valid; incomplete configurations restart at epoch 1. Prior completed summaries are retained in the aggregate results.

```powershell
python scripts/run_experiments.py `
  --phase 2 `
  --data .\data\luna16 `
  --cache-dir .\data\patches_cache_v3 `
  --train-subsets 1 `
  --validation-subset 0 `
  --out .\experiments_phase2 `
  --epochs 5 `
  --batch-size 4 `
  --max-configs 20 `
  --resume
```

Phase 2 writes per-configuration files under the output directory and aggregate `all_results.json` and `all_results.csv` files. Do not run a large grid until the data split, cache, checkpoint space, and compute budget are understood.

### Phase 3: Optuna search

```powershell
python scripts/run_experiments.py `
  --phase 3 `
  --data .\data\luna16 `
  --cache-dir .\data\patches_cache_v3 `
  --train-subsets 1 `
  --validation-subset 0 `
  --out .\experiments_phase3 `
  --epochs 30 `
  --n-trials 100
```

### Short Phase 1 smoke run

Use a unique output location and a tiny epoch count to check that the selected cache loads and training starts. **Phase 1 runs all six ablations**, each for one epoch in this example; it is not a single-configuration run. This still performs real model training and requires a valid cache and adequate memory:

```powershell
python scripts/run_experiments.py `
  --phase 1 `
  --data .\data\luna16 `
  --cache-dir .\data\patches_cache_v3_smoke `
  --train-subsets 1 `
  --validation-subset 0 `
  --out .\experiments_smoke `
  --epochs 1 `
  --batch-size 1
```

The CLI also accepts `--fold` (legacy fold split), `--test-subset` (only when a genuinely separate subset is available), `--lr`, and `--resume` (Phase 2). Requested train/validation/test subsets must exist and be disjoint by scan UID. The project does not invent a test subset when only two subsets are available.

## Model, losses, and metrics

`training/model.py` wraps Torchvision Mask R-CNN with a ResNet-50 Feature Pyramid Network backbone, a two-class background/nodule detector, configured anchor sizes, and a custom composite segmentation-mask loss. Inputs are three-channel 2.5D image patches. The raw loader extracts adjacent axial slices, uses physical image geometry for world/voxel conversion, and pads patches at volume boundaries while keeping the annotation center aligned. Positive masks are spherical/elliptical approximations derived from annotation center and diameter. They are pseudo-labels, not manual contours.

The composite mask objective supports these weights/components:

```text
L = α·BCE + β·Dice + γ·Boundary + δ·Focal + ε·HD_DT
```

`HD_DT` is a distance-transform surrogate; it is not an exact differentiable HD95. Phase 2 currently searches only `α`, `β`, and `γ`; Phase 3 samples `α`, `β`, `γ`, `δ`, and `ε`. Phase 1 evaluates the five supported loss terms. The web experiment form exposes the five weights `α` through `ε`.

Validation includes Dice, IoU, HD95, boundary F1, sensitivity, specificity, average surface distance, and volume error where defined. The training evaluator also reports sampled-negative-patch false-positive rate and sampled-patch detections-per-scan proxy. Those are patch-sampling diagnostics, not official scan-level false positives per scan. A metric can be unavailable on a dataset that lacks the cases needed to define it.

### Checkpoints and experiment files

Each run directory may contain `summary.json`, `history.json`, `terminal.log`, `best_model.pth`, and `latest_model.pth`. Checkpoints include the model state and training configuration; best checkpoints are selected on validation Dice. Training artifacts are local and ignored by Git. The FastAPI experiment page stores its record and epoch metrics in the configured SQL database, and logs/artifacts in the configured experiment directory.

## Evaluation

Evaluation requires an explicit fold. If a checkpoint records cache metadata, pass the same cache path and patch size to avoid silently evaluating a different preprocessing pipeline:

```powershell
python scripts/evaluate_model.py `
  --checkpoint .\experiments_phase2\G-a03-b05-g05\best_model.pth `
  --data .\data\luna16 `
  --fold 0 `
  --cache-dir .\data\patches_cache_v3 `
  --out .\eval_results
```

To compare multiple checkpoints, use `--checkpoints` followed by paths or glob patterns. The evaluator loads checkpoint architecture metadata and model weights strictly. It rejects unavailable subsets and mismatched configured cache paths. If no cache is supplied, it uses raw-volume loading and warns that the input distribution may differ from cache-based training.

Evaluation reports patch-based segmentation metrics and a **sampled-patch FROC proxy**. It does not implement official scan-level, candidate-matched LUNA16 FROC. Do not describe the proxy as official scan-level FP/scan.

## Web application

The React app is served at `/` and has these pages:

| Route | Page | Main data/actions |
| --- | --- | --- |
| `/` | Dashboard | Runtime health, experiment summary, epoch plots, persisted activity, reports |
| `/experiments` | Experiments | Create, start, inspect, monitor, and remove database experiment records |
| `/leaderboard` | Leaderboard | Filter and compare persisted experiments, validation metrics, Pareto view |
| `/inference` | Inference | Upload an exported RGB 64×64 2.5D patch; inspect detections and history |
| `/edge` | Edge Network | Registered nodes and persisted network measurements, with prototype limitations |
| `/dataset-model` | Dataset & Models | Cache validation summary, default model readiness, experiment checkpoints |

The dashboard polls summary, health, activity, cache, and system-metric endpoints every 30 seconds. Experiment progress uses the existing WebSocket and falls back to controlled polling when the connection is unavailable. Null or unsupported hardware metrics are shown as unavailable. CSV and JSON exports use persisted experiment values; charts can be saved as PNG, and the report can be printed or saved to PDF by the browser.

Inference accepts PNG/JPEG files that decode as true RGB images of exactly 64×64 pixels. The three channels are expected to correspond to axial slices `z−1`, `z`, and `z+1`; the API checks image mode and dimensions but cannot verify the clinical meaning or ordering of the channels. Select an experiment checkpoint explicitly when desired. Predictions are research output, not diagnosis.

## HTTP and WebSocket API

Interactive request/response schemas are available at <http://localhost:8000/docs> after startup.

### Health and dashboard

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | API and default inference-engine startup status |
| `GET` | `/api/info` | Project metadata and runtime dataset/cache summary |
| `GET` | `/api/dashboard/health` | Database, model, device/GPU, cache, and heartbeat status |
| `GET` | `/api/dashboard/summary` | Experiment counts, best-epoch metrics, inference/network counts |
| `GET` | `/api/dashboard/activity?limit=50` | Recent events derived from persisted rows |
| `GET` | `/api/dashboard/dataset` | Active path, split, and cache-validation information |
| `GET` | `/api/dashboard/system-metrics` | Available host/RAM and CUDA memory readings; unsupported fields are null |

### Experiments and results

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/experiments/` | Validate and create a pending experiment record/configuration |
| `GET` | `/api/experiments/` | List experiment records; accepts `phase`, `status`, `limit`, `offset` |
| `GET` | `/api/experiments/{id}` | Get one experiment |
| `GET` | `/api/experiments/{id}/details` | Configuration, run history, and captured terminal log |
| `GET` | `/api/experiments/{id}/runs` | Epoch metrics |
| `GET` | `/api/experiments/{id}/summary` | Experiment summary |
| `GET` | `/api/experiments/{id}/logs` | Captured terminal log |
| `POST` | `/api/experiments/{id}/start` | Start background training |
| `DELETE` | `/api/experiments/{id}` | Remove a non-running experiment record and related database run rows (local files are separate) |
| `WS` | `/api/experiments/{id}/ws` | Training progress updates |
| `GET` | `/api/results/stats` | Aggregate counts and best validation Dice |
| `GET` | `/api/results/leaderboard` | Best validation-Dice epoch metrics; optional `phase` and `status` filters |
| `GET` | `/api/results/pareto` | Validation Dice vs. HD95 Pareto set; optional filters |
| `GET` | `/api/results/experiment/{id}/learning-curve` | Epoch learning curve |

The experiment API and CLI support `loss_weights` (`alpha` BCE, `beta` Dice, `gamma` Boundary, `delta` Focal, `epsilon` distance-transform Hausdorff surrogate), `cache_mode`, `cache_dir`, `patch_size`, epochs, batch size, learning rate, and optional explicit subset split.

Example development request (replace the paths and split with files that actually exist on the backend host):

```json
{
  "config_id": "smoke-bce-dice-boundary",
  "phase": 1,
  "loss_weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.5, "delta": 0, "epsilon": 0},
  "epochs": 1,
  "batch_size": 1,
  "lr": 0.005,
  "cache_mode": "cached",
  "cache_dir": "D:/lung-nodule/data/patches_cache_v3_smoke",
  "data_root": "D:/lung-nodule/data/luna16",
  "train_subsets": [1],
  "validation_subset": 0
}
```

Send that JSON to `POST /api/experiments/`, then start the returned record ID with `POST /api/experiments/{id}/start`. This validates local data/cache paths and starts actual model training; use a development database and do not send a training request just to check API reachability.

### Inference and edge

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/inference/status` | Default checkpoint load state |
| `POST` | `/api/inference/upload` | Multipart PNG/JPEG 2.5D inference; optional experiment checkpoint |
| `POST` | `/api/inference/predict` | Base64 image request |
| `GET` | `/api/inference/results` | Persisted inference history |
| `POST` | `/api/edge/nodes/register` | Register/update an edge node |
| `GET` | `/api/edge/nodes` | List registered edge nodes |
| `POST` | `/api/edge/nodes/{node_id}/heartbeat` | Update last-seen timestamp |
| `POST` | `/api/edge/network-probe` | Consume a network probe request |
| `POST` | `/api/edge/network-log` | Persist a network observation |
| `GET` | `/api/edge/network-log` | List recorded observations |
| `GET` | `/api/edge/network-stats` | Aggregate recorded network measurements |
| `GET` | `/api/edge/split-latency-analysis` | Aggregate split-layer records |
| `POST` | `/api/edge/infer-features` | Authenticated intermediate-feature endpoint; current edge/cloud contract returns `501` for incompatible features |

## Edge and cloud prototype

Start the edge node with the backend reachable at `CLOUD_API_URL`. It registers, sends periodic heartbeats, and exposes its local HTTP service (default port `9090`). Configuration is through environment variables; see [Configuration reference](#configuration-reference).

```powershell
Set-Location edge
$env:CLOUD_API_URL = "http://localhost:8000"
$env:NODE_ID = "edge-node-01"
$env:SPLIT_LAYER = "layer2"
$env:COMPRESSION = "int8"
..\venv\Scripts\python.exe edge_node.py
```

```powershell
# Generate local development fixtures (not LUNA16), then simulate on one patch.
python scripts/generate_test_data.py --out .\data\synthetic_luna --patch-dir .\data\sample_patches --n-scans 2 --n-nodules 1 --subsets 2
$sample = (Get-ChildItem .\data\sample_patches\*_patch.png | Select-Object -First 1).FullName
python scripts/simulate_edge.py --image $sample --split layer2 --compress int8
```

The edge service provides `GET http://localhost:9090/health` and `POST http://localhost:9090/scan?scan_uid=<uid>`. The scan route accepts encoded patch bytes and returns JSON; when cloud completion is unavailable it attempts a local threshold-based candidate heuristic, not a trained Mask R-CNN detector. It is a prototype workflow, not a production PACS/DICOM integration.

The `cn1_split_analysis.py` script uses modeled LAN/Wi-Fi/4G/5G bandwidth and RTT profiles. Its output is a simulation/profile, not a measurement of a deployed carrier or hospital network. Backend network dashboards summarize records actually posted to the database. An absent row means no measurement was recorded; it is not treated as zero latency.

| CN objective | Current code/evaluation state |
| --- | --- |
| Split-point optimization | Profiling/simulation scripts and measurement endpoints exist; the extractor/cloud feature-map mismatch blocks successful cloud split inference. No best split has been established. |
| Feature compression | Local compressor supports `none`, `int8`, and an optional learned bottleneck; the current edge wire encoding serializes float32 values, so INT8 wire savings are not established. |
| Federated learning | Flower simulation and client/server modes exist; this does not demonstrate training across real hospitals. |
| Adaptive scheduling | A network heuristic exists, but automatic selection is not wired into the scan inference workflow. |
| Differential privacy | A standalone experimental script exists; it is not a live mechanism or a validated end-to-end privacy guarantee. |

## Other research scripts

- `scripts/cn1_split_analysis.py`: compare candidate split layers with configured network conditions and a checkpoint/data input.
- `training/federated.py`: Flower simulation/server/client modes using scan subsets as client partitions. A simulation does not establish deployment across independent hospitals.
- `scripts/cn5_differential_privacy.py`: experimental feature-noise and membership-inference privacy/utility analysis. This standalone script is not integrated into live edge inference and its assumptions must be reviewed before making a formal differential-privacy claim.
- `scripts/plot_results.py`: plot experiment/evaluation results.
- `scripts/import_experiment_folders.py`: import existing experiment output folders into the backend database.
- `scripts/generate_test_data.py`: generate synthetic development data; it is not LUNA16 and must not be reported as clinical benchmark data.

Use each script’s `--help` to inspect its current arguments before running it. CN profiling, federated learning, and privacy experiments may have separate dependencies or assumptions from the web application.

## Tests and checks

From the repository root with the project Python environment active:

```powershell
python -m pytest tests/ -v
```

The suite covers API behavior, cache validation and subset integrity, patch extraction boundaries, losses/metrics, checkpoint loading, experiment resume, edge payloads, and inference/training smoke cases. Tests use generated or synthetic fixtures and do not download the LUNA16 dataset. Some model-forward tests require a supported Torch/Torchvision installation and can be slow on CPU.

Frontend checks:

```powershell
Set-Location frontend
npm ci
npm run lint
npm run build
```

There is no separate frontend test script configured in `frontend/package.json` at this time. The lint command uses the repository’s ESLint flat config.

## Configuration reference

| Variable | Used by | Default / meaning |
| --- | --- | --- |
| `DATABASE_URL` | Backend | `sqlite:///./lung_nodule.db`; SQLite by default. Other SQLAlchemy dialects require the matching database driver to be installed. |
| `DATA_ROOT` | Backend/training | `data/luna16` (resolved relative to the repository by backend path helpers) |
| `LUNA16_CACHE_DIR` | Backend | `data/patches_cache_v3_smoke` |
| `MODEL_PATH` | Backend inference | `experiments/best_model.pth`; must point to an actual compatible checkpoint to load a default model |
| `EXPERIMENTS_DIR` | Backend checkpoint discovery | `experiments/`; HTTP training artifacts are stored under the repository `experiments/` directory |
| `PATCH_SIZE` | Backend | `64` |
| `PATCH_SPACING_MM` | Backend inference geometry | `1.0` |
| `SCORE_THRESHOLD` | Backend inference | `0.05` |
| `NMS_THRESHOLD` | Backend inference | `0.5` |
| `APP_ENV` | Backend edge authentication | Defaults to `production`; an absent `EDGE_API_TOKEN` is allowed only for explicit `dev`, `development`, `test`, or `testing` environments |
| `EDGE_API_TOKEN` | Edge feature endpoint/node | Shared secret; required outside explicit development/test mode and rejects invalid credentials |
| `EDGE_MODEL_PATH` | Edge node | Must point to the same trained checkpoint loaded by backend `MODEL_PATH`; Compose mounts `experiments/` read-only |
| `EDGE_HEARTBEAT_TIMEOUT_SECONDS` | Dashboard status | `60` seconds for dashboard online status |
| `CORS_ALLOW_ORIGINS` | Backend | `http://localhost:3000,http://127.0.0.1:3000` |
| `CORS_ALLOW_CREDENTIALS` | Backend | `false` |
| `CLOUD_API_URL` | Edge node | `http://localhost:8000` |
| `NODE_ID` | Edge node | `edge-node-01` |
| `SPLIT_LAYER` | Edge node | `layer2` |
| `COMPRESSION` | Edge node | `int8` |
| `HEARTBEAT_SEC` | Edge node | `30` seconds |
| `LEARNED_COMPRESSOR_CHECKPOINT` | Edge learned compressor | Optional checkpoint path; learned compression needs a trained compatible checkpoint |
Docker Compose defaults `APP_ENV` to `production`; set it explicitly to `development` only for a local setup that intentionally has no edge token. The edge extractor loads the checkpoint at `EDGE_MODEL_PATH`, and split inference verifies its SHA-256 against the backend checkpoint and checks that edge/cloud input normalization matches. Keep `EDGE_MODEL_PATH` and `MODEL_PATH` pointed at the same file. Docker Compose reads the root `.env`; a manually launched Python process does not automatically read `.env` in this code. Export/set the needed variables in the shell before launching a local process.

The FastAPI app creates its tables at startup. An initial Alembic migration is also checked in under `backend/alembic/versions/`; set `DATABASE_URL` before invoking Alembic for a deployment migration workflow.

## Data, checkpoints, and Git

`.gitignore` excludes `data/`, model weights (`.pth`, `.ckpt`, `.pt`, `.h5`), experiment outputs, SQLite databases, logs, `frontend/node_modules/`, and `.env`. Keep raw scans and cache contents in local storage; do not commit the LUNA16 dataset, patient-derived files, credentials, or large checkpoints. The repository does not automatically upload runtime experiment results to GitHub.

The backend database contains `experiments`, `training_runs`, `inference_results`, `edge_nodes`, and `network_logs`. Checkpoints and captured run logs are files under the experiment output directory. Deleting a non-running experiment removes its database record and related epoch rows; it does not promise to remove experiment files. Back up the database and experiment files together if you need to preserve a run.

## Research scope and limitations

- **Pseudo masks:** masks are approximated from nodule center and diameter; they are not expert-drawn contours. Segmentation scores measure agreement with generated masks.
- **Limited split:** development data may contain only `subset0` and `subset1`. With one used for training and one for validation, there is no independent test set. Do not describe validation as test performance.
- **Patch sampling:** training and evaluation are patch-based. Negative examples are sampled patches, not a verified clinical negative cohort. Current false-positive fields are explicitly patch-level proxies.
- **Negative background approximation:** the raw loader searches crop-safe non-air voxels and excludes physical neighborhoods around annotations; this intensity-based sampling is not a lung segmentation mask. Cache generation prefers labeled class-0 candidate locations when available, but it does not turn patch labels into expert contours.
- **FROC:** the evaluation script does not perform official scan-level candidate matching; its FROC-like summary is a sampled-patch proxy.
- **Spacing and preprocessing:** new cache generation resamples scans to the configured target spacing (default 1 mm isotropic). Older caches may have different preprocessing/spacing metadata. Do not combine inconsistent caches in a comparison.
- **Edge/cloud feature contract:** the edge feature extractor and backend continuation currently expect incompatible feature maps. The feature endpoint validates/authenticates input but returns HTTP `501` for the current extractor payload; that request does not produce a cloud prediction.
- **Compression claims:** some feature serialization transmits float32 values even when the local compressor reports int8. The UI labels ratios as unverified. Do not claim wire-level savings until transmitted-byte measurement confirms them.
- **Scheduling:** network-based scheduling logic is not integrated into automatic inference-path selection.
- **Federated learning:** Flower code supports simulation/client/server workflows; local LUNA16 partitions do not represent ten real hospitals or establish multi-institution deployment.
- **Differential privacy:** the DP script is a standalone experimental analysis, not a live privacy guarantee for the web/edge pipeline. Formal claims require validated sensitivity, privacy accounting, and the deployed mechanism.
- **System telemetry:** CPU/RAM and CUDA memory are returned only when the host dependencies/runtime expose them. GPU utilization and temperature may be unavailable; the dashboard does not fabricate values.
- **Clinical use:** no clinical validation, regulatory clearance, radiologist workflow, or diagnostic guarantee is provided.

## References

- He et al., [Mask R-CNN](https://arxiv.org/abs/1703.06870), 2017.
- Kervadec et al., [Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032), 2019.
- Setio et al., [Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: the LUNA16 challenge](https://doi.org/10.1016/j.media.2017.06.011), 2017.
- Karimi and Salcudean, [Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks](https://arxiv.org/abs/1904.10030), 2019.
- McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629), 2017.
