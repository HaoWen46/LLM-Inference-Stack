## LLM Stack — top-level Makefile
##
## Usage:
##   make setup         — install Python deps with uv
##   make download      — pull model weights
##   make vllm          — start vLLM server
##   make llamacpp      — start llama-cpp-python OpenAI server (GGUF models)
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
CARGO     := $(shell which cargo 2>/dev/null || echo $(HOME)/.cargo/bin/cargo)
RUST_MANIFEST := $(ROOT_DIR)/rust/Cargo.toml

.PHONY: help setup download vllm llamacpp gateway gpu-exporter build-rust \
        db monitoring monitoring-down loadtest status stop logs

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

# ── llama-cpp-python server (GGUF models) ──────────────────────────────────
llamacpp:
	@bash scripts/launch_llamacpp.sh

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

# ── PostgreSQL (API key + quota store) ─────────────────────────────────────
db:
	@docker compose -f docker/docker-compose.yml up -d postgres
	@echo "PostgreSQL: localhost:5432  (user: gateway)"

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
# API key resolution (in order of priority):
#   1. API_KEY=<key> make loadtest   (explicit, recommended with DB-backed keys)
#   2. First entry in GATEWAY_API_KEYS from config/.env  (legacy seed key)
#   3. Abort with a clear error if neither is set
loadtest:
	@{ \
	  set -a; source $(ENV_FILE); set +a; \
	  KEY="$${API_KEY:-$$(echo "$${GATEWAY_API_KEYS:-}" | cut -d, -f1)}"; \
	  if [ -z "$$KEY" ]; then \
	      echo "[loadtest] ERROR: no API key."; \
	      echo "  Pass API_KEY=<key> make loadtest, or set GATEWAY_API_KEYS in config/.env"; \
	      exit 1; \
	  fi; \
	  PORT="$${GATEWAY_PORT:-8080}"; \
	  if ! curl -sf "http://localhost:$$PORT/health" >/dev/null 2>&1; then \
	      echo "[loadtest] ERROR: gateway not responding on :$$PORT — is it running?"; \
	      exit 1; \
	  fi; \
	  $(UV) run loadtest/runner.py \
	      --url "http://localhost:$$PORT" \
	      --key "$$KEY" \
	      --model "$${SERVED_MODEL_NAME:-$$MODEL_NAME}" \
	      --concurrency "$${CONCURRENCY:-16}" \
	      --duration "$${DURATION:-60}"; \
	}

# ── Status ─────────────────────────────────────────────────────────────────
status:
	@echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "  (no GPU)"
	@echo ""
	@echo "=== Processes ===" && ps aux | grep -E 'vllm|gateway|gpu-exporter' | grep -v grep || true
	@echo ""
	@{ set -a; source $(ENV_FILE); set +a; \
	   VLLM_PORT=$${VLLM_PORT:-8000}; \
	   GATEWAY_PORT=$${GATEWAY_PORT:-8080}; \
	   echo "=== vLLM health (port $$VLLM_PORT) ===" && curl -sf http://localhost:$$VLLM_PORT/health && echo " OK" || echo " DOWN"; \
	   echo "=== Gateway health (port $$GATEWAY_PORT) ===" && curl -sf http://localhost:$$GATEWAY_PORT/health && echo " OK" || echo " DOWN"; \
	   echo "=== GPU exporter ===" && curl -sf http://localhost:9101/health && echo " OK" || echo " DOWN"; \
	}

# ── Stop all (SIGTERM → wait up to 35s → SIGKILL) ─────────────────────────
# Safety: never use `pkill -f <pattern>` — the pattern literal lives in the
# shell's argv and causes self-termination.  Instead:
#   • Rust binaries: pgrep -x <name>  (match by comm, not full cmdline)
#   • vLLM main:     pgrep -f with [.] trick so literal string ≠ regex match
#   • vLLM workers:  pgrep <VLLM>     (match comm prefix, no -f)
stop:
	@echo "Stopping services..."
	@{ \
	  _kill() { sig=$$1; shift; pids="$$*"; [ -n "$$pids" ] && kill -$$sig $$pids 2>/dev/null || true; }; \
	  _kill TERM $$(pgrep -x gateway 2>/dev/null); \
	  _kill TERM $$(pgrep -x gpu-exporter 2>/dev/null); \
	  _kill TERM $$(pgrep -f 'vllm[.]entrypoints' 2>/dev/null); \
	  _kill TERM $$(pgrep 'VLLM' 2>/dev/null); \
	  _kill TERM $$(pgrep -f 'llama_cpp[.]server' 2>/dev/null); \
	  printf 'Waiting for graceful shutdown'; \
	  end=$$((SECONDS + 35)); \
	  while \
	      pgrep -x gateway >/dev/null 2>&1 || \
	      pgrep -f 'vllm[.]entrypoints' >/dev/null 2>&1 || \
	      pgrep 'VLLM' >/dev/null 2>&1 || \
	      pgrep -f 'llama_cpp[.]server' >/dev/null 2>&1; do \
	      [ $$SECONDS -ge $$end ] && break; \
	      printf '.'; sleep 1; \
	  done; echo; \
	  _kill KILL $$(pgrep -x gateway 2>/dev/null); \
	  _kill KILL $$(pgrep -x gpu-exporter 2>/dev/null); \
	  _kill KILL $$(pgrep -f 'vllm[.]entrypoints' 2>/dev/null); \
	  _kill KILL $$(pgrep 'VLLM' 2>/dev/null); \
	  _kill KILL $$(pgrep -f 'llama_cpp[.]server' 2>/dev/null); \
	}
	@echo "Stopped."

# ── Log tailing ────────────────────────────────────────────────────────────
logs:
	@docker compose -f docker/docker-compose.yml logs -f loki promtail
