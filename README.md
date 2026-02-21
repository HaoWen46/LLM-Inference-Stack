# LLM Inference Stack

Production-grade LLM serving on 2x NVIDIA RTX A6000 (48GB each).

Built on [vLLM](https://github.com/vllm-project/vllm) with a **Rust/Axum gateway**, Prometheus metrics, and Grafana dashboards.

## Architecture

```
Client
  │
  ▼
NGINX  (TLS termination, optional)
  │
  ▼
Gateway :8080  (Rust/Axum — auth, rate-limit, quota, circuit breaker, metrics)
  │
  ▼
vLLM   :8000  (inference, KV cache, tensor parallelism)
  │
  ├── GPU 0  RTX A6000 48GB
  └── GPU 1  RTX A6000 48GB

Sidecar processes
  gpu-exporter :9101  →  Prometheus :9090  →  Grafana :3000
  (Rust binary — polls nvidia-smi every 2s)
```

## Stack

| Component | Language | Purpose |
|-----------|----------|---------|
| vLLM 0.6.6 | Python | Inference engine — paged attention, continuous batching, tensor parallelism |
| **Axum 0.7** | **Rust** | **API gateway — zero-GC, predictable tail latency** |
| **governor** | **Rust** | **Per-IP GCRA rate limiting** |
| **sqlx + DashMap** | **Rust** | **Per-key daily token quota (SQLite backend)** |
| **prometheus-client** | **Rust** | **Metrics exposition** |
| Prometheus 2.55 | — | Metrics storage |
| Grafana 11.3 | — | Dashboards and alerting |
| Jaeger | — | Distributed tracing (OpenTelemetry OTLP) |
| Loki + Promtail | — | Log aggregation |

The Python FastAPI gateway (`gateway/`) is kept as a reference implementation and fallback.

## Hardware requirements

| Component | Spec |
|-----------|------|
| GPU | 2x NVIDIA RTX A6000 (48GB VRAM each) |
| RAM | 128GB system |
| Storage | NVMe SSD recommended for model loading |
| OS | Ubuntu 22.04 |
| CUDA | 12.x |

## Quickstart

### 1. Install dependencies

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python deps (vLLM, load tester)
make setup

# Install Rust (for gateway + gpu-exporter)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Configure

```bash
# config/.env is created by `make setup` — edit before proceeding
vim config/.env
```

Key values to set:

```bash
MODEL_NAME=meta-llama/Llama-3.2-7B-Instruct
HF_TOKEN=hf_...              # required for gated models
VLLM_API_KEY=change-me       # internal secret between gateway and vLLM
GATEWAY_API_KEYS=key1,key2   # bearer tokens your clients will use
```

For a 7B model on a single GPU:
```bash
TP_SIZE=1
QUANTIZATION=none
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
```

For a 70B model across both GPUs (requires quantization to fit):
```bash
TP_SIZE=2
QUANTIZATION=awq             # ~35GB; fits with KV headroom
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=8192
```

### 3. Download model

```bash
make download
# or a specific model:
bash scripts/download_model.sh meta-llama/Llama-3.2-7B-Instruct
```

Models are saved to `./models/` (configurable via `MODEL_CACHE_DIR` in `.env`).

### 4. Run the stack

Open three terminal panes (or use tmux):

```bash
# Pane 1 — inference engine
make vllm

# Pane 2 — API gateway (Rust; builds on first run)
make gateway

# Pane 3 — GPU metrics exporter (Rust)
make gpu-exporter
```

The first `make gateway` invocation compiles the Rust binary (`~60s`). Subsequent runs start in milliseconds.

### 5. Start monitoring

Requires Docker for Prometheus + Grafana:

```bash
make monitoring
# Prometheus → http://localhost:9090
# Grafana    → http://localhost:3000   (admin / admin)
# Jaeger UI  → http://localhost:16686
```

The Grafana dashboard auto-provisions on first start.

### 6. Test

```bash
# Health check
curl http://localhost:8080/health

# Streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "What is tensor parallelism?"}],
    "max_tokens": 256,
    "stream": true
  }'

# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'

# Per-key usage
curl http://localhost:8080/v1/usage \
  -H "Authorization: Bearer dev-key-1"
```

### 7. Load test

```bash
make loadtest
# tune concurrency and duration:
CONCURRENCY=32 DURATION=120 make loadtest
```

Outputs live stats (req/s, P50/P95/P99 latency, TTFT) and a final report broken down by prompt type.

## File layout

```
.
├── config/
│   ├── .env                  ← your local config (gitignored)
│   └── .env.example          ← template
│
├── rust/                     ← Rust workspace (gateway + gpu-exporter)
│   ├── Cargo.toml            ← workspace root
│   ├── gateway/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs       ← router, warmup, graceful shutdown
│   │       ├── config.rs     ← Config::from_env()
│   │       ├── auth.rs       ← ApiKey extractor (Bearer + x-api-key)
│   │       ├── rate_limiter.rs ← DashMap<IpAddr, governor::RateLimiter>
│   │       ├── circuit_breaker.rs ← lockless atomics (no Mutex)
│   │       ├── quota.rs      ← DashMap cache + SQLite flush every 10s
│   │       ├── metrics.rs    ← prometheus-client Registry
│   │       ├── proxy.rs      ← sync + SSE streaming proxy, TTFT
│   │       ├── error.rs      ← GatewayError → HTTP status mapping
│   │       └── tracing_setup.rs ← tracing-subscriber JSON + OTel OTLP
│   └── gpu-exporter/
│       ├── Cargo.toml
│       └── src/
│           └── main.rs       ← nvidia-smi polling, /metrics, /health
│
├── gateway/                  ← Python gateway (reference / fallback)
│   ├── app.py                ← FastAPI app: proxy, streaming, auth, metrics
│   ├── auth.py
│   ├── circuit_breaker.py
│   ├── config.py
│   ├── logging_config.py
│   ├── metrics.py
│   ├── server.py
│   └── usage_db.py
│
├── scripts/
│   ├── launch_vllm.sh        ← starts vLLM with settings from .env
│   ├── download_model.sh     ← huggingface_hub snapshot_download
│   ├── gpu_exporter.py       ← Python GPU exporter (replaced by Rust)
│   └── watchdog.py           ← process supervisor with health checks
│
├── loadtest/
│   └── runner.py             ← async load driver, rich live output, percentile report
│
├── observability/
│   ├── prometheus/
│   │   ├── prometheus.yml    ← scrape config (vLLM + gateway + GPU exporter)
│   │   └── alerts.yml        ← KV cache pressure, latency, ECC, queue depth
│   ├── grafana/
│   │   ├── dashboards/       ← auto-provisioned LLM stack dashboard
│   │   └── provisioning/     ← datasource + dashboard provider config
│   ├── loki/                 ← Loki log aggregation config
│   └── promtail/             ← Promtail log shipper config
│
├── docker/
│   ├── Dockerfile.gateway-rust     ← multi-stage Rust build (~30MB image)
│   ├── Dockerfile.gpu-exporter-rust
│   ├── Dockerfile.gateway          ← Python gateway (fallback)
│   ├── Dockerfile.vllm
│   └── docker-compose.yml          ← full stack
│
├── pyproject.toml
└── Makefile
```

## Makefile targets

```
make setup          install Python deps into .venv via uv
make download       pull model weights from HuggingFace
make vllm           start vLLM server (foreground)
make gateway        build + start Rust gateway (foreground)
make gpu-exporter   build + start Rust GPU Prometheus exporter
make build-rust     compile both Rust crates (release, no run)
make monitoring     start Prometheus + Grafana + Jaeger via docker-compose
make loadtest       run load test against gateway
make status         show GPU state and process health
make stop           gracefully stop all background processes
make logs           tail Loki + Promtail container logs
```

## Rust gateway — design notes

| Feature | Implementation |
|---------|---------------|
| Rate limiting | Per-IP GCRA via `governor`; lazy `DashMap<IpAddr, Arc<RateLimiter>>` |
| Circuit breaker | Lockless `AtomicU8` state + `AtomicU64` failure counter; `compare_exchange` transitions |
| Quota | `DashMap` in-memory cache (atomic fast path) flushed to SQLite every 10s; SHA-256 key hashing |
| Streaming | `reqwest::bytes_stream()` → line buffer → `tokio::mpsc` → `Body::from_stream` |
| TTFT | Measured on first non-empty `data: ` SSE line; recorded as Prometheus histogram |
| Graceful shutdown | `SIGTERM`/`Ctrl-C` → `shutting_down` flag → 30s drain |
| Observability | `tracing-subscriber` JSON + optional OpenTelemetry OTLP to Jaeger |

## Memory budget

Quick reference for this hardware (96GB total VRAM):

| Model | Precision | Weights | KV headroom | Notes |
|-------|-----------|---------|-------------|-------|
| 7B | bf16 | ~14GB | ~82GB | 1 GPU, massive batch capacity |
| 13B | bf16 | ~26GB | ~70GB | 1 GPU |
| 70B | int8 | ~70GB | ~26GB | 2 GPUs, TP=2 |
| 70B | AWQ int4 | ~35GB | ~61GB | 2 GPUs, TP=2, recommended |
| Mixtral 8x7B | int8 | ~56GB | ~40GB | 2 GPUs, TP=2 |

KV cache per token (Llama-3 70B, GQA, fp16): ~320KB/token.
With 26GB headroom after int8 weights: ~83K tokens of total KV capacity.

## Observability

The Grafana dashboard covers:

- **Traffic** — req/s, active requests, queue depth
- **Latency** — P50/P95/P99 end-to-end, TTFT percentiles
- **GPU** — utilization, memory used/free, power draw, temperature, SM clock
- **KV cache** — GPU utilization %, CPU swap, running/waiting/swapped sequences
- **Errors** — 4xx/5xx by status code, auth failures, rate-limited requests, quota exceeded

Alert rules fire on:
- KV cache > 85% / 95%
- Requests being swapped to CPU
- P99 latency > 60s
- Queue depth > 50
- GPU memory < 2GB free
- GPU temperature > 85°C
- Any uncorrected ECC errors

## Gateway API

The gateway is OpenAI-compatible. Any client that works with the OpenAI SDK works here:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dev-key-1",
)

# Streaming
stream = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Explain paged attention."}],
    max_tokens=512,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Authentication accepts both `Authorization: Bearer <key>` and `X-API-Key: <key>` headers.

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | — | Liveness check |
| GET | `/ready` | — | Readiness (polls vLLM `/health`) |
| GET | `/metrics` | — | Prometheus text metrics |
| GET | `/v1/models` | ✓ | List available models |
| POST | `/v1/chat/completions` | ✓ | Chat completions (streaming or buffered) |
| POST | `/v1/completions` | ✓ | Legacy completions |
| GET | `/v1/usage` | ✓ | Per-key daily token usage |

## Production notes

**Do not expose vLLM directly.** It has no auth and minimal error handling. Always route through the gateway.

**NCCL hangs:** Set `TORCH_NCCL_BLOCKING_WAIT=1` (already in the launch script) so NCCL errors surface instead of hanging indefinitely. The watchdog in `scripts/watchdog.py` restarts processes that fail health checks.

**KV cache pressure:** If `vllm:gpu_cache_usage_perc` exceeds 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS` in `.env`. The alert fires at 85%.

**Cold start:** First request after startup is slow — vLLM compiles Triton kernels on first run and caches them in `.triton_cache/`. Subsequent starts are faster. The gateway's `/ready` endpoint blocks until vLLM is healthy.

**CPU swap:** `SWAP_SPACE_GB=8` in `.env` configures vLLM to spill KV blocks to RAM when GPU cache is full. Use as a safety valve only — PCIe bandwidth makes active swapping very slow.

**First Rust build:** `make gateway` compiles from source on first run (~60s). Use `make build-rust` to pre-compile, or build the Docker image (`docker build -f docker/Dockerfile.gateway-rust .`) for a reproducible artifact.
