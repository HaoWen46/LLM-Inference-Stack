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
make gateway        # builds + starts Rust/Axum gateway on :8080
make gpu-exporter   # builds + starts Rust GPU Prometheus exporter on :9101

# Monitoring stack (Docker)
make monitoring     # Prometheus :9090, Grafana :3000, Jaeger :16686
make monitoring-down

# Load test
make loadtest
CONCURRENCY=32 DURATION=120 make loadtest

# Status / stop
make status
make stop
```

The first `make gateway` compiles Rust from source (~60s). Subsequent runs start in milliseconds. There are no lint or unit test targets — this is an operational stack.

## Architecture

```
Client → NGINX (optional TLS) → Gateway :8080 (Rust/Axum)
                                     → vLLM :8000 → GPU 0 + GPU 1

Sidecar: gpu-exporter :9101 → Prometheus :9090 → Grafana :3000
         Jaeger :16686 (OTLP tracing)
         Loki + Promtail (log aggregation)
```

**Active hot path**: Rust/Axum gateway (`rust/gateway/`). The Python FastAPI gateway (`gateway/`) is kept as a reference implementation only.

## Configuration

All runtime config lives in `config/.env` (gitignored; template at `config/.env.example`). Key variables:

- `MODEL_NAME`, `HF_TOKEN`, `MODEL_CACHE_DIR` — model identity and weights path
- `VLLM_API_KEY` — internal secret between gateway and vLLM (not client-facing)
- `GATEWAY_API_KEYS` — comma-separated bearer tokens for clients
- `TP_SIZE`, `QUANTIZATION`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS` — vLLM performance tuning
- `SWAP_SPACE_GB` — KV block CPU spill safety valve (use sparingly; PCIe bandwidth makes active swapping slow)

The gateway reads all config via `Config::from_env()` at startup (`rust/gateway/src/config.rs`). No dotenv — variables must be exported in the shell or set in the launch environment.

## Rust Gateway — Key Design Points

The gateway is a single Rust workspace at `rust/` with two crates: `gateway` and `gpu-exporter`.

**gateway/src layout:**

| File | Role |
|------|------|
| `main.rs` | Axum router, warmup task, graceful shutdown (30s drain on SIGTERM) |
| `proxy.rs` | Core dispatcher: sync proxy + SSE streaming via `reqwest::bytes_stream()` → `tokio::mpsc` → `Body::from_stream`; TTFT measured on first `data:` line |
| `quota.rs` | Per-key daily token quota: `DashMap` in-memory fast path, flushed to SQLite every 10s; SHA-256 key hashing |
| `rate_limiter.rs` | Per-IP GCRA via `governor`; lazy `DashMap<IpAddr, Arc<RateLimiter>>` |
| `circuit_breaker.rs` | Lockless: `AtomicU8` state (CLOSED/OPEN/HALF_OPEN) + `AtomicU64` failure counter, `compare_exchange` transitions |
| `metrics.rs` | `prometheus-client` registry: request counters, latency histograms, TTFT, active gauge |
| `auth.rs` | `ApiKey` extractor — accepts `Authorization: Bearer <key>` or `X-API-Key: <key>` |
| `tracing_setup.rs` | JSON structured logging + optional OpenTelemetry OTLP to Jaeger |

**Do not expose vLLM directly** — it has no auth. Always route through the gateway.

## Observability Endpoints

| URL | Description |
|-----|-------------|
| `GET /health` | Liveness (no auth) |
| `GET /ready` | Readiness — polls vLLM `/health` |
| `GET /metrics` | Prometheus text format |
| `GET /v1/usage` | Per-key daily token usage (auth required) |

The gateway is OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).

## Operational Notes

- **KV cache pressure**: If `vllm:gpu_cache_usage_perc` > 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS`. Alerts fire at 85%.
- **Cold start**: vLLM compiles Triton kernels on first-ever run; results are cached in `.triton_cache/`. The gateway's `/ready` endpoint blocks until vLLM is healthy.
- **NCCL hangs**: `TORCH_NCCL_BLOCKING_WAIT=1` is set in `scripts/launch_vllm.sh` so errors surface instead of hanging. `scripts/watchdog.py` can supervise and restart failed processes.
- **Memory budget quick ref** (96GB total VRAM): 7B bf16 ≈ 14GB weights; 70B AWQ int4 ≈ 35GB weights (recommended for TP=2).
