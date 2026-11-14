# =============================================================================
# Lung Nodule Detection — Mask R-CNN Edge–Cloud Pipeline
# =============================================================================
# Usage:  make <target>
# =============================================================================

.PHONY: help setup setup-dev data backend frontend edge \
        docker docker-gpu docker-down \
        train-p1 train-p2 train-p3 \
        evaluate federated dp \
        test test-metrics test-api test-losses \
        lint clean

PYTHON      ?= python3
DATA_ROOT   ?= ./data/luna16
EXP_OUT     ?= ./experiments
EVAL_OUT    ?= ./eval_results
FOLD        ?= 0
TEST_FOLD   ?= 9
EPOCHS      ?= 30
BATCH       ?= 4
CHECKPOINT  ?= $(EXP_OUT)/best_model.pth

# ─── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🫁  Lung Nodule — Mask R-CNN Edge–Cloud Pipeline"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup          Install all Python dependencies"
	@echo "    make setup-dev      + dev/test tools (pytest, optuna, flwr)"
	@echo "    make data           Generate synthetic LUNA16 test data"
	@echo ""
	@echo "  SERVICES"
	@echo "    make backend        Run FastAPI backend (port 8000)"
	@echo "    make frontend       Run React dashboard (port 3000)"
	@echo "    make edge           Run edge node service (port 9090)"
	@echo "    make docker         Build & launch all services (Docker)"
	@echo "    make docker-gpu     Same with GPU support"
	@echo "    make docker-down    Stop all containers"
	@echo ""
	@echo "  TRAINING (DIP Track)"
	@echo "    make train-p1       Phase 1 — single-loss ablations"
	@echo "    make train-p2       Phase 2 — grid search (α,β,γ)"
	@echo "    make train-p3       Phase 3 — Bayesian optimisation"
	@echo "    make evaluate       Evaluate all checkpoints on test fold"
	@echo ""
	@echo "  RESEARCH (CN Track)"
	@echo "    make federated      Run federated learning simulation (CN3)"
	@echo "    make dp             Run differential privacy experiment (CN5)"
	@echo "    make simulate-edge  Simulate edge–cloud pipeline on a sample image"
	@echo ""
	@echo "  TESTS"
	@echo "    make test           Run all tests"
	@echo "    make test-losses    Run loss function tests only"
	@echo "    make test-metrics   Run metric tests only"
	@echo "    make test-api       Run API integration tests"
	@echo "    make lint           Run flake8 linter"
	@echo ""
	@echo "  OTHER"
	@echo "    make clean          Remove .pyc files and __pycache__"
	@echo ""

# ─── Setup ───────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -r training/requirements.txt
	$(PYTHON) -m pip install -r backend/requirements.txt
	$(PYTHON) -m pip install -r edge/requirements.txt
	cd frontend && npm install

setup-dev: setup
	$(PYTHON) -m pip install pytest pytest-asyncio httpx optuna flwr scikit-learn
	@echo "Dev tools installed."

# ─── Data ────────────────────────────────────────────────────────────────────
data:
	@echo "Generating synthetic LUNA16 test data..."
	$(PYTHON) scripts/generate_test_data.py \
	  --out $(DATA_ROOT) \
	  --n-scans 20 \
	  --n-nodules 3 \
	  --subsets 2

data-real:
	@echo "Download LUNA16 from https://luna16.grand-challenge.org/Download/"
	@echo "Place subset0..9/, annotations.csv, candidates.csv in: $(DATA_ROOT)"

# ─── Services ────────────────────────────────────────────────────────────────
backend:
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

edge:
	cd edge && $(PYTHON) edge_node.py

# ─── Docker ──────────────────────────────────────────────────────────────────
docker:
	docker compose up --build

docker-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ─── Training — DIP Track ────────────────────────────────────────────────────
train-p1:
	@echo "=== Phase 1: Single-loss ablations ==="
	PYTHONPATH=. $(PYTHON) scripts/run_experiments.py \
	  --phase 1 \
	  --data $(DATA_ROOT) \
	  --out $(EXP_OUT) \
	  --epochs $(EPOCHS) \
	  --batch-size $(BATCH) \
	  --fold $(FOLD)

train-p2:
	@echo "=== Phase 2: Grid search (alpha, beta, gamma) ==="
	PYTHONPATH=. $(PYTHON) scripts/run_experiments.py \
	  --phase 2 \
	  --data $(DATA_ROOT) \
	  --out $(EXP_OUT) \
	  --epochs $(EPOCHS) \
	  --batch-size $(BATCH) \
	  --fold $(FOLD) \
	  --max-configs 40

train-p3:
	@echo "=== Phase 3: Bayesian optimisation (Optuna) ==="
	PYTHONPATH=. $(PYTHON) scripts/run_experiments.py \
	  --phase 3 \
	  --data $(DATA_ROOT) \
	  --out $(EXP_OUT) \
	  --epochs $(EPOCHS) \
	  --batch-size $(BATCH) \
	  --fold $(FOLD) \
	  --n-trials 100

evaluate:
	@echo "=== Evaluating all checkpoints on fold $(TEST_FOLD) ==="
	PYTHONPATH=. $(PYTHON) scripts/evaluate_model.py \
	  --checkpoints $(EXP_OUT)/*/best_model.pth \
	  --data $(DATA_ROOT) \
	  --fold $(TEST_FOLD) \
	  --out $(EVAL_OUT)

evaluate-single:
	PYTHONPATH=. $(PYTHON) scripts/evaluate_model.py \
	  --checkpoint $(CHECKPOINT) \
	  --data $(DATA_ROOT) \
	  --fold $(TEST_FOLD) \
	  --out $(EVAL_OUT)

# ─── Research — CN Track ─────────────────────────────────────────────────────
federated:
	@echo "=== CN3: Federated Learning simulation ==="
	PYTHONPATH=. $(PYTHON) -m training.federated \
	  --mode simulate \
	  --data $(DATA_ROOT) \
	  --num-clients 5 \
	  --rounds 20 \
	  --strategy fedavg \
	  --out $(EXP_OUT)/federated

federated-fedprox:
	PYTHONPATH=. $(PYTHON) -m training.federated \
	  --mode simulate \
	  --data $(DATA_ROOT) \
	  --num-clients 5 \
	  --rounds 20 \
	  --strategy fedprox \
	  --out $(EXP_OUT)/federated_fedprox

dp:
	@echo "=== CN5: Differential Privacy experiment ==="
	PYTHONPATH=. $(PYTHON) scripts/cn5_differential_privacy.py \
	  --checkpoint $(CHECKPOINT) \
	  --data $(DATA_ROOT) \
	  --split layer2 \
	  --epsilons 0.1 0.5 1.0 5.0 10.0 50.0 \
	  --out $(EXP_OUT)/cn5_dp

simulate-edge:
	@$(eval SAMPLE := $(shell ls data/sample_patches/*.png 2>/dev/null | head -1))
	@if [ -z "$(SAMPLE)" ]; then \
	  echo "No sample patches found. Run 'make data' first."; exit 1; fi
	PYTHONPATH=. $(PYTHON) scripts/simulate_edge.py \
	  --image $(SAMPLE) \
	  --split layer2 \
	  --compress int8

simulate-edge-cloud:
	@$(eval SAMPLE := $(shell ls data/sample_patches/*.png 2>/dev/null | head -1))
	PYTHONPATH=. $(PYTHON) scripts/simulate_edge.py \
	  --image $(SAMPLE) \
	  --split layer2 \
	  --compress int8 \
	  --cloud http://localhost:8000

# ─── Tests ───────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --tb=short

test-losses:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_losses.py -v --tb=short

test-metrics:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_metrics.py -v --tb=short

test-api:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_api.py -v --tb=short

test-fast:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

lint:
	$(PYTHON) -m flake8 training/ backend/ edge/ scripts/ \
	  --max-line-length=120 \
	  --ignore=E501,W503,E203 \
	  --exclude=__pycache__,*.pyc

# ─── Utilities ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	@echo "Cleaned."

clean-experiments:
	rm -rf $(EXP_OUT)/*
	@echo "Experiments cleared."

clean-data:
	rm -rf $(DATA_ROOT)
	@echo "Data cleared."

# ─── Quick start ─────────────────────────────────────────────────────────────
quickstart: setup data
	@echo ""
	@echo "Quick start complete. Run 'make docker' to launch all services, or:"
	@echo "  Terminal 1:  make backend"
	@echo "  Terminal 2:  make frontend"
	@echo "  Terminal 3:  make edge"
	@echo "  Then:        make train-p1 EPOCHS=5"
