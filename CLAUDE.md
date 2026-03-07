# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Python deps (via uv)
make setup

# Compile both Rust crates (no run)
make build-rust
# cargo build --manifest-path rust/Cargo.toml --release

# Run individual services (foreground)
make vllm           # starts vLLM inference server
make llamacpp       # starts llama-cpp-python server for GGUF models on :8001
make gateway        # builds + starts Rust/Axum gateway on :8080
make gpu-exporter   # builds + starts Rust GPU Prometheus exporter on :9101

# PostgreSQL (required by gateway — API key store + quota)
make db             # docker compose up postgres (localhost:5432)

# Monitoring stack (Docker)
make monitoring     # Prometheus :9090, Grafana :3000, Jaeger :16686, Loki :3100
make monitoring-down

# Load test (pass API key explicitly when using DB-backed keys)
API_KEY=gw_... make loadtest
CONCURRENCY=32 DURATION=120 API_KEY=gw_... make loadtest

# Status / stop
make status
make stop
```

The first `make gateway` compiles Rust from source (~60s). Subsequent runs start in milliseconds. There are no lint or unit test targets — this is an operational stack.

## Architecture

```
Client → NGINX (optional TLS) → Gateway :8080 (Rust/Axum)
                                     → vLLM :8000 → GPU 0 + GPU 1  (Qwen3.5-27B, TP=2)

Sidecar: gpu-exporter :9101 → Prometheus :9090 → Grafana :3000
         Jaeger :16686 (OTLP tracing)
         Loki + Promtail (log aggregation)
```

**Active hot path**: Rust/Axum gateway (`rust/gateway/`). The Python FastAPI gateway (`gateway/`) is kept as a reference implementation only.

## Configuration

All runtime config lives in `config/.env` (gitignored; template at `config/.env.example`). Key variables:

- `MODEL_NAME`, `HF_TOKEN`, `MODEL_CACHE_DIR` — model identity and weights path
- `VLLM_API_KEY` — internal secret between gateway and vLLM; required if `BACKENDS` is not set
- `BACKENDS` — multi-backend config (optional); format: `url|api_key|models|weight|served_name` semicolon-separated; if not set, falls back to `VLLM_URL` + `VLLM_API_KEY` + `SERVED_MODEL_NAME` as a single backend
- `DATABASE_URL` — PostgreSQL connection string (required by gateway)
- `ADMIN_KEY` — Bearer token for `/admin/*` key management endpoints
- `GATEWAY_API_KEYS` — optional seed: if set and the DB has no keys, each entry is auto-inserted once at startup; use the admin API for ongoing key management
- `TP_SIZE`, `QUANTIZATION`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS` — vLLM performance tuning
- `SWAP_SPACE_GB` — KV block CPU spill safety valve (use sparingly; PCIe bandwidth makes active swapping slow)

The gateway reads all config via `Config::from_env()` at startup (`rust/gateway/src/config.rs`). No dotenv — variables must be exported in the shell or set in the launch environment.

## Rust Gateway — Key Design Points

The gateway is a single Rust workspace at `rust/` with two crates: `gateway` and `gpu-exporter`.

**gateway/src layout:**

| File | Role |
|------|------|
| `main.rs` | Axum router, warmup task (polls primary backend `/health`), graceful shutdown (30s drain on SIGTERM) |
| `backends.rs` | `BackendDef`/`Backend`/`BackendRouter`: per-backend CB + `AtomicBool` health; weighted random selection; `start_health_task()` polls every 10s |
| `proxy.rs` | Core dispatcher: `backend_router.select(model)` → sync/SSE proxy; `models_handler` fans out to all healthy backends; model-name rewrite via `backend.served_model_name` |
| `auth.rs` | `ApiKey` extractor — accepts `Authorization: Bearer <key>` or `X-API-Key: <key>`; SHA-256 hash → DashMap lookup; checks `expires_at` on hot path |
| `keys.rs` | `KeyStore`: PgPool + DashMap cache; key generation (`gw_` prefix), CRUD, background refresh every 60s; org CRUD and org usage aggregation |
| `admin.rs` | Admin REST handlers: `/admin/keys` CRUD + `PUT /admin/keys/:id/limits`; `/admin/orgs` CRUD + `GET /admin/orgs/:id/usage` |
| `quota.rs` | Per-key daily token quota: `DashMap` in-memory fast path, flushed to PostgreSQL every 10s; `is_allowed()` accepts per-key limit override |
| `rate_limiter.rs` | Per-IP GCRA via `governor`; separate per-key DashMap for `rpm_limit` enforcement; background eviction of idle entries |
| `batches.rs` | Batch inference API — `POST/GET /v1/batches`, `GET/POST /:id`, `GET /:id/results`; background tokio worker routes each item via `backend_router`; stored in `batches` table |
| `circuit_breaker.rs` | Lockless: `AtomicU8` state (CLOSED/OPEN/HALF_OPEN) + `AtomicU64` failure counter, `compare_exchange` transitions; one instance per backend |
| `metrics.rs` | `prometheus-client` registry: request counters, latency histograms, TTFT, active gauge |
| `tracing_setup.rs` | JSON structured logging + optional OpenTelemetry OTLP to Jaeger |

**Do not expose vLLM directly** — it has no auth. Always route through the gateway.

## Observability Endpoints

| URL | Auth | Description |
|-----|------|-------------|
| `GET /health` | — | Liveness (no auth) |
| `GET /ready` | — | Readiness — polls vLLM `/health` |
| `GET /metrics` | — | Prometheus text format |
| `GET /v1/usage` | API key | Per-key daily token usage (shows effective quota: per-key or global) |
| `GET /v1/models` | API key | List models loaded in vLLM |
| `GET /v1/models/local` | API key | Scan `MODEL_CACHE_DIR` for HF dirs and GGUF files |
| `POST /v1/chat/completions` | API key | Chat completions (streaming or buffered) |
| `POST /v1/completions` | API key | Legacy completions |
| `POST /v1/embeddings` | API key | Embeddings proxy (no model-name rewrite) |
| `POST /v1/tokenize` | API key | Tokenize passthrough → vLLM `/tokenize` |
| `POST /v1/detokenize` | API key | Detokenize passthrough → vLLM `/detokenize` |
| `POST /v1/batches` | API key | Create offline batch job (inline requests) |
| `GET /v1/batches` | API key | List batch jobs for this key |
| `GET /v1/batches/:id` | API key | Poll batch status |
| `POST /v1/batches/:id/cancel` | API key | Cancel a batch job |
| `GET /v1/batches/:id/results` | API key | Fetch completed batch results |
| `POST /admin/keys` | Admin key | Create API key — plaintext shown once |
| `GET /admin/keys` | Admin key | List all keys with metadata and limits |
| `PATCH /admin/keys/:id` | Admin key | Enable or disable a key |
| `PUT /admin/keys/:id/limits` | Admin key | Replace per-key limits (rpm, daily tokens, allowlist, expiry, org) |
| `DELETE /admin/keys/:id` | Admin key | Delete a key permanently |
| `POST /admin/orgs` | Admin key | Create organisation |
| `GET /admin/orgs` | Admin key | List organisations |
| `GET /admin/orgs/:id/usage` | Admin key | Aggregate token usage for all keys in an org |

## Operational Notes

- **KV cache pressure**: If `vllm:gpu_cache_usage_perc` > 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS`. Alerts fire at 85%.
- **Cold start**: vLLM compiles Triton kernels on first-ever run; results are cached in `.triton_cache/`. The gateway's `/ready` endpoint blocks until vLLM is healthy.
- **NCCL hangs**: `TORCH_NCCL_BLOCKING_WAIT=1` is set in `scripts/launch_vllm.sh` so errors surface instead of hanging. `scripts/watchdog.py` can supervise and restart failed processes.
- **Memory budget quick ref** (144GB total VRAM — 3× RTX A6000 48GB): 7B bf16 ≈ 14GB weights; 35B bf16 ≈ 70GB weights (TP=2, ~33GB/GPU + KV headroom); 70B AWQ int4 ≈ 35GB weights.
- **BF16 GEMM fix**: torch 2.10.0 + `nvidia-cublas-cu12==12.8.4.1` + CUDA driver 12.9 breaks all BF16 GEMMs on Ampere (CC 8.6). `launch_vllm.sh` auto-upgrades to `12.9.1.4` on every launch (idempotent). If you need to fix it manually: `uv pip install "nvidia-cublas-cu12==12.9.1.4" --no-deps`.
- **vLLM nightly rotation**: the nightly wheel server only keeps the latest build. If `uv sync` fails with "version not found", update the vllm pin in `pyproject.toml` to the current nightly version from `wheels.vllm.ai/nightly`.
- **`make stop` safety**: never use `pkill -f <pattern>` in scripts — the pattern literal lives in the shell's argv and causes self-termination. Use `pgrep -x <name>` for Rust binaries, `pgrep -f 'vllm[.]entrypoints'` (the `[.]` trick), and `pgrep 'VLLM'` for worker processes.
