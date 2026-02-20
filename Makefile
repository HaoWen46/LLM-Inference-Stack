## LLM Stack — top-level Makefile
##
## Usage:
##   make setup        — create .env and install deps
##   make download     — pull model weights
##   make vllm         — start vLLM server
##   make gateway      — start API gateway
##   make gpu-exporter — start GPU Prometheus exporter
##   make monitoring   — start Prometheus + Grafana via docker-compose
##   make loadtest     — run load test against gateway
##   make status       — show running processes and GPU state
##   make stop         — stop all background processes

SHELL     := bash
ROOT_DIR  := $(shell pwd)
VENV      := $(ROOT_DIR)/.venv
PYTHON    := $(VENV)/bin/python3
UV        := $(shell which uv || echo $(HOME)/.local/bin/uv)
ENV_FILE  := $(ROOT_DIR)/config/.env

.PHONY: help setup download vllm gateway gpu-exporter monitoring loadtest status stop

help:
	@grep -E '^## ' Makefile | sed 's/## //'

# ── Setup ──────────────────────────────────────────────────────────────────
setup: $(VENV)/bin/activate config/.env

$(VENV)/bin/activate: pyproject.toml
	@echo "[setup] Creating venv and installing deps..."
	$(UV) venv --python 3.10 $(VENV)
	$(UV) pip install --python $(PYTHON) -r pyproject.toml
	@touch $(VENV)/bin/activate
	@echo "[setup] Done. Activate with: source .venv/bin/activate"

config/.env:
	@echo "[setup] Creating config/.env from example..."
	cp config/.env.example config/.env
	@echo "[setup] Edit config/.env before proceeding."

# ── Model download ─────────────────────────────────────────────────────────
download:
	@bash scripts/download_model.sh

# ── vLLM server (runs in foreground — use tmux or & for background) ────────
vllm:
	@bash scripts/launch_vllm.sh

# ── API Gateway ────────────────────────────────────────────────────────────
gateway:
	@source $(ENV_FILE) && \
	  source $(VENV)/bin/activate && \
	  PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m gateway.server

# ── GPU metrics exporter ───────────────────────────────────────────────────
gpu-exporter:
	@source $(VENV)/bin/activate && \
	  $(PYTHON) scripts/gpu_exporter.py

# ── Monitoring stack (Prometheus + Grafana) ────────────────────────────────
monitoring:
	@docker compose -f docker/docker-compose.yml up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000  (admin / admin)"

monitoring-down:
	@docker compose -f docker/docker-compose.yml down

# ── Load test ──────────────────────────────────────────────────────────────
loadtest:
	@source $(VENV)/bin/activate && \
	  source $(ENV_FILE) && \
	  PYTHONPATH=$(ROOT_DIR) $(PYTHON) loadtest/runner.py \
	    --url http://localhost:8080 \
	    --key $$(echo $$GATEWAY_API_KEYS | cut -d, -f1) \
	    --model $$SERVED_MODEL_NAME \
	    --concurrency $${CONCURRENCY:-16} \
	    --duration $${DURATION:-60}

# ── Status ─────────────────────────────────────────────────────────────────
status:
	@echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader
	@echo ""
	@echo "=== Processes ===" && ps aux | grep -E 'vllm|gateway|gpu_exporter' | grep -v grep || true
	@echo ""
	@echo "=== vLLM health ===" && curl -sf http://localhost:8000/health && echo " OK" || echo " DOWN"
	@echo "=== Gateway health ===" && curl -sf http://localhost:8080/health && echo " OK" || echo " DOWN"

# ── Stop all ───────────────────────────────────────────────────────────────
stop:
	@pkill -f 'vllm.entrypoints' || true
	@pkill -f 'gateway.server'   || true
	@pkill -f 'gpu_exporter'     || true
	@echo "Stopped."
