# LLM Inference Stack

Production-grade LLM serving on 4× NVIDIA RTX 4090 + 2× RTX 3090 (144 GB total VRAM).

Built on [vLLM](https://github.com/vllm-project/vllm) with a **Rust/Axum gateway**, Prometheus metrics, and Grafana dashboards.

**Current deployment:** `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` on GPUs 1 & 2 (RTX 4090, TP=2).

## Architecture

```
Client
  │
  ▼
NGINX  (TLS termination, optional)
  │
  ▼
Gateway :8020  (Rust/Axum — auth, rate-limit, quota, circuit breaker, metrics)
  │
  ▼
vLLM   :8010  (inference, KV cache, tensor parallelism)
  │
  ├── GPU 1  RTX 4090 24GB
  └── GPU 2  RTX 4090 24GB

Sidecar processes
  gpu-exporter :9101  →  Prometheus :9090  →  Grafana :3000
  (Rust binary — polls nvidia-smi every 2s)
```

## Stack

| Component | Language | Purpose |
|-----------|----------|---------|
| vLLM 0.10.0 | Python | Inference engine — paged attention, continuous batching, tensor parallelism |
| **Axum 0.7** | **Rust** | **API gateway — zero-GC, predictable tail latency** |
| **governor** | **Rust** | **Per-IP GCRA rate limiting** |
| **sqlx + DashMap** | **Rust** | **Per-key daily token quota (SQLite backend)** |
| **prometheus-client** | **Rust** | **Metrics exposition** |
| Prometheus 2.55 | — | Metrics storage |
| Grafana 11.3 | — | Dashboards and alerting |
| Jaeger | — | Distributed tracing (OpenTelemetry OTLP) |
| Loki + Promtail | — | Log aggregation |

The Python FastAPI gateway (`gateway/`) is kept as a reference implementation and fallback.

## Hardware

| # | GPU | VRAM | Notes |
|---|-----|------|-------|
| 0 | NVIDIA RTX 4090 | 24 GB | available |
| 1 | NVIDIA RTX 4090 | 24 GB | **active — Qwen3-30B shard 0** |
| 2 | NVIDIA RTX 4090 | 24 GB | **active — Qwen3-30B shard 1** |
| 3 | NVIDIA RTX 4090 | 24 GB | available |
| 4 | NVIDIA RTX 3090 | 24 GB | available |
| 5 | NVIDIA RTX 3090 | 24 GB | available |

- **RAM:** 128 GB system
- **OS:** Linux (6.12 LTS kernel)
- **CUDA:** 12.x
- **Python:** 3.12 (managed via [uv](https://github.com/astral-sh/uv))
- **Rust:** 1.8x (stable)

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

> **uv cache:** if your home partition is small, redirect the cache:
> `export UV_CACHE_DIR=/fast-disk/.cache/uv`
>
> **vLLM compile cache:** vLLM writes `torch_compile_cache` to `~/.cache/vllm` by
> default. On machines with a small home quota this causes an `OSError: Disk quota
> exceeded` crash. Set `VLLM_CACHE_ROOT` in `.env` to redirect it:
> `VLLM_CACHE_ROOT=/fast-disk/.cache/vllm`

### 2. Configure

```bash
cp config/.env.example config/.env
vim config/.env
```

Key values:

```bash
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
SERVED_MODEL_NAME=qwen3-30b
CUDA_VISIBLE_DEVICES=1,2    # which GPUs to use

VLLM_PORT=8010
VLLM_URL=http://localhost:8010   # gateway reads this, not VLLM_PORT
VLLM_API_KEY=change-me           # internal secret between gateway and vLLM

GATEWAY_PORT=8020
GATEWAY_API_KEYS=key1,key2       # bearer tokens your clients will use

TP_SIZE=2                        # one shard per GPU
DTYPE=auto                       # auto-detects FP8 from model config.json
QUANTIZATION=none                # leave none; vLLM handles FP8 natively
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=32
```

For a 7B bf16 model on a single GPU:

```bash
TP_SIZE=1
DTYPE=bfloat16
QUANTIZATION=none
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=256
```

For a 70B AWQ model across two GPUs:

```bash
TP_SIZE=2
QUANTIZATION=awq
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=8192
```

### 3. Download model

```bash
make download
# or explicitly:
bash scripts/download_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

Models are saved to `./models/` (configurable via `MODEL_CACHE_DIR` in `.env`).

### 4. Run the stack

Open three terminal panes (or use tmux):

```bash
# Pane 1 — inference engine (via uv run)
set -a && source config/.env && set +a
UV_CACHE_DIR=/fast-disk/.cache/uv \
uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --api-key "$VLLM_API_KEY" \
  --enable-prefix-caching --enable-chunked-prefill

# Pane 2 — API gateway (Rust; builds on first run ~60s)
make gateway

# Pane 3 — GPU metrics exporter (Rust)
make gpu-exporter
```

Alternatively, `make vllm` calls `scripts/launch_vllm.sh` which activates the project `.venv` directly.

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
# Liveness
curl http://localhost:8020/health

# Readiness (polls vLLM /health)
curl http://localhost:8020/ready

# Non-streaming chat
curl http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "What is tensor parallelism?"}],
    "max_tokens": 256
  }'

# Streaming chat
curl http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "Explain MoE briefly."}],
    "max_tokens": 256,
    "stream": true
  }'

# Per-key token usage
curl http://localhost:8020/v1/usage \
  -H "Authorization: Bearer dev-key-1"
```

### 7. Load test

```bash
make loadtest
# tune concurrency and duration:
CONCURRENCY=32 DURATION=120 make loadtest
```

Outputs live stats (req/s, P50/P95/P99 latency, TTFT) and a final report.

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
│   ├── app.py
│   ├── auth.py
│   ├── circuit_breaker.py
│   ├── config.py
│   ├── metrics.py
│   └── usage_db.py
│
├── scripts/
│   ├── launch_vllm.sh        ← starts vLLM with settings from .env (uses .venv)
│   ├── download_model.sh     ← huggingface_hub snapshot_download
│   ├── write_kernel_configs.py ← generate heuristic Triton configs for RTX 4090 (instant, no GPU)
│   ├── tune_kernels.py       ← benchmark-tune FP8/MoE Triton configs on a free GPU
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
│   ├── loki/
│   └── promtail/
│
├── docker/
│   ├── Dockerfile.gateway-rust     ← multi-stage Rust build (~30MB image)
│   ├── Dockerfile.gpu-exporter-rust
│   ├── Dockerfile.gateway          ← Python gateway (fallback)
│   ├── Dockerfile.vllm
│   └── docker-compose.yml
│
├── pyproject.toml
└── Makefile
```

## Makefile targets

```
make setup          install Python deps into .venv via uv
make download       pull model weights from HuggingFace
make vllm           start vLLM (foreground, uses .venv)
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
| Authentication | `Authorization: Bearer` or `X-API-Key`; constant-time comparison via `subtle` crate (no timing side-channels) |
| Rate limiting | Per-IP GCRA via `governor`; lazy `DashMap<IpAddr, Arc<RateLimiter>>`; background task evicts entries idle >1 hour |
| Circuit breaker | Lockless `AtomicU8` state + `AtomicU64` failure counter; `compare_exchange` transitions |
| Quota | `DashMap` in-memory cache (atomic fast path) flushed to SQLite every 10s; SHA-256 key hashing |
| Request size limit | `DefaultBodyLimit::max(4 MB)` — oversized bodies rejected with 413 before parsing or auth |
| Streaming | `reqwest::bytes_stream()` → line buffer → `tokio::mpsc` → `Body::from_stream`; injects `stream_options: {include_usage: true}` for exact token counts |
| TTFT | Measured on first non-empty `data: ` SSE line; recorded as Prometheus histogram |
| Token counting | Streaming: exact `prompt_tokens`/`completion_tokens` from vLLM's final usage chunk; sync: from `usage` field in response JSON |
| Security headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection` on every response via `SetResponseHeaderLayer` |
| Error sanitization | `UpstreamError` messages (IPs, socket details) logged internally; clients receive only `"Upstream service unavailable"` |
| Graceful shutdown | `SIGTERM`/`Ctrl-C` → `shutting_down` flag → 30s drain |
| Observability | `tracing-subscriber` JSON + optional OpenTelemetry OTLP to Jaeger |

## Memory budget

Reference for RTX 4090 (24 GB VRAM each):

| Model | Precision | Weights/GPU (TP=2) | KV headroom | Notes |
|-------|-----------|-------------------|-------------|-------|
| Qwen3-30B-A3B | FP8 | ~14.6 GB | ~7 GB | **current; 2× RTX 4090** |
| 7B | bf16 | ~14 GB | ~8 GB | single GPU |
| 13B | bf16 | ~13 GB | ~9 GB | TP=2 |
| 70B | AWQ int4 | ~17.5 GB | ~5 GB | TP=2, tight |
| Mixtral 8×7B | int8 | ~21 GB | ~2 GB | TP=2, very tight |

> **FP8 on RTX 4090:** vLLM ships Triton kernel configs only for data-center GPUs
> (H100, A100, etc.). Run `scripts/write_kernel_configs.py` (instant, no GPU needed)
> to generate heuristic configs that suppress the "sub-optimal" startup warnings, or
> run `scripts/tune_kernels.py` (5–10 min on a free GPU) for benchmarked-optimal configs.
>
> **Known hardware limits on RTX 4090 (not fixable):**
> - `CutlassBlockScaledGroupedGemm` — requires Hopper tensor memory accelerator; silent fallback to standard GEMM.
> - FlashInfer sampling — no pre-built wheel for torch 2.7.x yet; PyTorch-native top-k/top-p fallback is functionally equivalent.

KV cache size depends on `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, and head dimensions. Check
`vllm:gpu_kv_cache_usage_perc` in Grafana — alert fires at 85%.

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
- GPU memory < 2 GB free
- GPU temperature > 85°C
- Any uncorrected ECC errors

## Gateway API

The gateway is OpenAI-compatible. Any client that works with the OpenAI SDK works here:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="dev-key-1",
)

# Streaming
stream = client.chat.completions.create(
    model="qwen3-30b",
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

**API key security:** generate keys with `openssl rand -hex 32` and set them in `GATEWAY_API_KEYS`. The gateway warns at startup if placeholder keys (`dev-key-1`, `dev-key-2`) are still configured. Keys are compared in constant time via the `subtle` crate and stored as SHA-256 hashes in SQLite.

**NCCL hangs:** `TORCH_NCCL_BLOCKING_WAIT=1` is set in `scripts/launch_vllm.sh` so NCCL errors surface instead of hanging. `scripts/watchdog.py` can supervise and restart failed processes.

**KV cache pressure:** If `vllm:gpu_cache_usage_perc` exceeds 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS` in `.env`. Alert fires at 85%.

**CUDA device order:** With a mixed GPU fleet (RTX 4090 + RTX 3090), set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (or use `CUDA_VISIBLE_DEVICES` explicitly) to ensure consistent device indexing.

**Cold start:** vLLM compiles Triton kernels on first-ever run; cached in `.triton_cache/`. The gateway's `/ready` endpoint blocks until vLLM is healthy. Model load + graph capture for Qwen3-30B-A3B-FP8 takes ~3–4 minutes.

**CPU swap:** `SWAP_SPACE_GB=8` configures vLLM to spill KV blocks to RAM when GPU cache is full. Use as a safety valve only — PCIe bandwidth makes active swapping very slow.

**First Rust build:** `make gateway` compiles from source on first run (~60s). Use `make build-rust` to pre-compile.

**Disk quota on home partition:** vLLM 0.10+ writes `torch_compile_cache` to `~/.cache/vllm` during the first warmup pass. On machines with a small home quota this causes an `OSError: [Errno 122] Disk quota exceeded` crash mid-startup. Set `VLLM_CACHE_ROOT` in `.env` to redirect all vLLM caches to a partition with more space:
```bash
VLLM_CACHE_ROOT=/fast-disk/.cache/vllm
```

**Triton kernel configs for consumer GPUs:** vLLM ships pre-tuned FP8/MoE Triton configs only for data-center GPUs. On RTX 4090 (or similar), run once after install:
```bash
# Instant heuristic configs — suppresses "sub-optimal" startup warnings
uv run python scripts/write_kernel_configs.py

# Optional: benchmark-tuned configs (5–10 min, run on a free GPU)
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  uv run python scripts/tune_kernels.py
```
