## LLM Stack — top-level Makefile
##
## Usage:
##   make setup         — install Python deps with uv
##   make download      — pull model weights
##   make vllm          — start vLLM server
##   make gateway       — start Rust gateway (builds first)
##   make gpu-exporter  — start Rust GPU Prometheus exporter
##   make build-rust    — compile both Rust crates (release)
##   make monitoring    — start Prometheus + Grafana + Jaeger + Loki via docker-compose
##   make loadtest      — run load test against gateway
##   make status        — show running processes and GPU state
##   make stop          — gracefully stop all background processes
##   make logs          — tail Loki + Promtail container logs

SHELL     := bash
ROOT_DIR  := $(shell pwd)
UV        := $(shell which uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
ENV_FILE  := $(ROOT_DIR)/config/.env
CARGO     := cargo
RUST_MANIFEST := $(ROOT_DIR)/rust/Cargo.toml

.PHONY: help setup download vllm gateway gpu-exporter build-rust \
        monitoring monitoring-down loadtest status stop logs

help:
	@grep -E '^## ' Makefile | sed 's/## //'

# ── Setup (Python deps via uv) ──────────────────────────────────────────────
setup: config/.env
	@echo "[setup] Installing Python deps with uv..."
	$(UV) sync
	@echo "[setup] Done. Run 'source .venv/bin/activate' or prefix commands with 'uv run'."

config/.env:
	@echo "[setup] Creating config/.env from example..."
	cp config/.env.example config/.env
	@echo "[setup] Edit config/.env before proceeding."

# ── Model download ─────────────────────────────────────────────────────────
download:
	@bash scripts/download_model.sh

# ── vLLM server ────────────────────────────────────────────────────────────
vllm:
	@bash scripts/launch_vllm.sh

# ── Rust gateway (builds then runs) ────────────────────────────────────────
gateway:
	@source $(ENV_FILE) && \
	  $(CARGO) run --manifest-path $(RUST_MANIFEST) -p gateway --release

# ── Rust GPU exporter ───────────────────────────────────────────────────────
gpu-exporter:
	@source $(ENV_FILE) && \
	  $(CARGO) run --manifest-path $(RUST_MANIFEST) -p gpu-exporter --release

# ── Build both Rust crates ──────────────────────────────────────────────────
build-rust:
	$(CARGO) build --manifest-path $(RUST_MANIFEST) --release
	@echo "[build-rust] Binaries at rust/target/release/{gateway,gpu-exporter}"

# ── Monitoring stack ────────────────────────────────────────────────────────
monitoring:
	@docker compose -f docker/docker-compose.yml up -d prometheus grafana jaeger loki promtail
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000  (admin / admin)"
	@echo "Jaeger UI:  http://localhost:16686"
	@echo "Loki:       http://localhost:3100"

monitoring-down:
	@docker compose -f docker/docker-compose.yml down

# ── Load test ──────────────────────────────────────────────────────────────
loadtest:
	@source $(ENV_FILE) && \
	  $(UV) run loadtest/runner.py \
	    --url http://localhost:8080 \
	    --key $$(echo $$GATEWAY_API_KEYS | cut -d, -f1) \
	    --model $$SERVED_MODEL_NAME \
	    --concurrency $${CONCURRENCY:-16} \
	    --duration $${DURATION:-60}

# ── Status ─────────────────────────────────────────────────────────────────
status:
	@echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "  (no GPU)"
	@echo ""
	@echo "=== Processes ===" && ps aux | grep -E 'vllm|gateway|gpu.exporter' | grep -v grep || true
	@echo ""
	@echo "=== vLLM health ===" && curl -sf http://localhost:8000/health && echo " OK" || echo " DOWN"
	@echo "=== Gateway health ===" && curl -sf http://localhost:8080/health && echo " OK" || echo " DOWN"
	@echo "=== GPU exporter ===" && curl -sf http://localhost:9101/health && echo " OK" || echo " DOWN"

# ── Stop all (SIGTERM → 35s → SIGKILL) ────────────────────────────────────
stop:
	@echo "Sending SIGTERM to gateway..."
	@pkill -SIGTERM -f 'target/release/gateway' || true
	@echo "Sending SIGTERM to vLLM..."
	@pkill -SIGTERM -f 'vllm.entrypoints' || true
	@pkill -SIGTERM -f 'target/release/gpu-exporter' || true
	@echo "Waiting up to 35s for graceful shutdown..."
	@sleep 35
	@pkill -9 -f 'vllm.entrypoints'         2>/dev/null || true
	@pkill -9 -f 'target/release/gateway'   2>/dev/null || true
	@pkill -9 -f 'target/release/gpu-exporter' 2>/dev/null || true
	@echo "Stopped."

# ── Log tailing ────────────────────────────────────────────────────────────
logs:
	@docker compose -f docker/docker-compose.yml logs -f loki promtail
