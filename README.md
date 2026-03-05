# LLM Inference Stack

Production-grade LLM serving on 3× NVIDIA RTX A6000 (144 GB total VRAM).

Built on [vLLM](https://github.com/vllm-project/vllm) with a **Rust/Axum gateway**, Prometheus metrics, and Grafana dashboards.

**Current deployment:** `Qwen/Qwen3-30B-A3B` (BF16) on GPUs 0 & 1 (A6000, TP=2).

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
| vLLM 0.16.0 | Python | Inference engine — paged attention, continuous batching, tensor parallelism |
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
| 0 | NVIDIA RTX A6000 | 48 GB | **active — Qwen3-30B shard 0** |
| 1 | NVIDIA RTX A6000 | 48 GB | **active — Qwen3-30B shard 1** |
| 2 | NVIDIA RTX A6000 | 48 GB | available |

- **RAM:** 128 GB system
- **OS:** Linux (6.8 kernel)
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
# If /tmp is mounted noexec (common on cluster nodes), rustup unpacking fails.
# Work around it by pointing TMPDIR at a writable directory first:
#   mkdir -p ~/.tmp && TMPDIR=~/.tmp sh rustup-init.sh -y
```

> **vLLM compile cache:** vLLM writes `torch_compile_cache` to `~/.cache/vllm` by
> default. On machines with a small home quota this causes an `OSError: Disk quota
> exceeded` crash. `VLLM_CACHE_ROOT` in `.env` (defaults to `.cache/vllm` inside the
> project) redirects it automatically.
>
> **Model storage on NFS:** If your home directory is NFS-mounted, loading model
> weights over the network is slow (10–15 min for a 60GB model vs ~15s from local
> NVMe). Store models on a local disk — e.g. `/tmp/<user>/models` — and set
> `MODEL_CACHE_DIR` and `MODEL_NAME` accordingly. `/tmp` on this machine is a 100GB
> tmpfs backed by RAM, which is fast but does not survive reboots.

### 2. Configure

```bash
cp config/.env.example config/.env
vim config/.env
```

Key values:

```bash
MODEL_NAME=/tmp/B11902156/models/Qwen/Qwen3-30B-A3B
SERVED_MODEL_NAME=Qwen/Qwen3-30B-A3B
MODEL_CACHE_DIR=/tmp/B11902156/models   # scanned by GET /v1/models/local

VLLM_PORT=8000
VLLM_API_KEY=change-me                  # internal secret between gateway and vLLM

GATEWAY_PORT=8080
GATEWAY_API_KEYS=key1,key2             # bearer tokens your clients will use

TP_SIZE=2                              # one shard per GPU
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=256
```

For a 7B bf16 model on a single GPU:

```bash
TP_SIZE=1
DTYPE=bfloat16
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
# or explicitly (full HF repo):
bash scripts/download_model.sh Qwen/Qwen3-30B-A3B

# single GGUF file (bypasses huggingface_hub / hf_xet):
bash scripts/download_model.sh TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q8_0.gguf
```

Models are saved to `MODEL_CACHE_DIR/<repo-id>/` (configurable via `.env`).

> **Model compatibility:** vLLM 0.16.0 requires `transformers<5`. Very new model
> architectures (e.g. `qwen3_5_moe`, `glm4_moe_lite`) only land in transformers 5.x
> and require the vLLM nightly wheel. Use established architectures (Qwen3, Llama,
> Mistral, DeepSeek-R1-Distill) with the stable release.

### 4. Run the stack

Open two terminal panes (or use tmux):

```bash
# Pane 1 — inference engine
make vllm
# or directly:
bash scripts/launch_vllm.sh

# Pane 2 — API gateway (Rust; builds on first run ~60s)
make gateway
```

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
curl http://localhost:8080/health

# Readiness (polls vLLM /health)
curl http://localhost:8080/ready

# Non-streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "What is tensor parallelism?"}],
    "max_tokens": 256
  }'

# Streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "Explain MoE briefly."}],
    "max_tokens": 256,
    "stream": true
  }'

# Per-key token usage
curl http://localhost:8080/v1/usage \
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
│   ├── launch_vllm.sh        ← starts vLLM with settings from .env (uses uv run)
│   ├── download_model.sh     ← HF snapshot_download (uv run) or single-file wget for GGUF
│   ├── write_kernel_configs.py ← generate heuristic Triton configs (instant, no GPU)
│   ├── tune_kernels.py       ← benchmark-tune Triton configs on a free GPU
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
├── tests/
│   ├── conftest.py           ← fixtures: mock vLLM (respx), ASGI transport, DB stubs
│   └── test_openai_compat.py ← 17 OpenAI SDK compatibility tests
│
├── pyproject.toml
└── Makefile
```

## Makefile targets

```
make setup          install Python deps into .venv via uv
make download       pull model weights from HuggingFace
make vllm           start vLLM (foreground, via scripts/launch_vllm.sh)
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

Reference for RTX A6000 (48 GB VRAM each):

| Model | Precision | Weights/GPU (TP=2) | KV headroom | Notes |
|-------|-----------|-------------------|-------------|-------|
| Qwen3-30B-A3B | bf16 | ~28 GB | ~18 GB | **current; 2× A6000** |
| 7B | bf16 | ~7 GB | ~39 GB | single GPU |
| 13B | bf16 | ~13 GB | ~35 GB | single GPU |
| 70B | bf16 | ~35 GB | ~13 GB | TP=2 |
| 70B | AWQ int4 | ~17.5 GB | ~28 GB | TP=2 |

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
    base_url="http://localhost:8080/v1",
    api_key="dev-key-1",
)

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
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
| GET | `/v1/models` | ✓ | List models currently loaded in vLLM |
| GET | `/v1/models/local` | ✓ | List model weight files on disk (see below) |
| POST | `/v1/chat/completions` | ✓ | Chat completions (streaming or buffered) |
| POST | `/v1/completions` | ✓ | Legacy completions |
| GET | `/v1/usage` | ✓ | Per-key daily token usage |

### `/v1/models/local` — discover models on disk

Returns every GGUF file and HuggingFace model directory found under `MODEL_CACHE_DIR`, formatted so you can copy-paste the `id` straight into your request body.

```bash
curl http://localhost:8080/v1/models/local \
  -H "Authorization: Bearer dev-key-1"
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-30B-A3B",
      "object": "model",
      "owned_by": "local",
      "format": "hf"
    }
  ]
}
```

Scans two layouts:
- **GGUF files** — any `*.gguf` found recursively; `id` is the path relative to `MODEL_CACHE_DIR`
- **HuggingFace Hub cache** — `models--<org>--<repo>` directories created by `huggingface_hub`
- **Flat org/repo dirs** — plain `<org>/<repo>` directories (e.g. `/tmp/user/models/Qwen/Qwen3-30B-A3B`) detected by checking for `config.json` in subdirectories

### Error format

All errors use the OpenAI error schema so standard SDK exception handling works without modification:

```json
{
  "error": {
    "message": "Invalid or missing API key",
    "type": "authentication_error",
    "param": null,
    "code": null
  }
}
```

| HTTP status | `type` |
|-------------|--------|
| 401 | `authentication_error` |
| 400 / 422 | `invalid_request_error` |
| 429 | `rate_limit_error` |
| 5xx | `api_error` |

## Testing

The test suite validates OpenAI SDK compatibility against the Python gateway running in-process — no GPU or vLLM process required.

```bash
# Install dev deps only (skips vllm/torch)
uv sync --only-group dev --no-install-project

# Run all 17 tests
uv run pytest tests/ -v
```

Tests cover: `/health`, `/v1/models`, `/v1/models/local`, chat completions (sync + streaming), authentication errors, invalid JSON, upstream 5xx passthrough, OpenAI error schema validation, and per-key usage endpoint.

vLLM is mocked with `respx`; the gateway's SQLite usage-DB is patched with `AsyncMock` stubs so tests are fully self-contained.

## Production notes

**Do not expose vLLM directly.** It has no auth and minimal error handling. Always route through the gateway.

**API key security:** generate keys with `openssl rand -hex 32` and set them in `GATEWAY_API_KEYS`. The gateway warns at startup if placeholder keys (`dev-key-1`, `dev-key-2`) are still configured. Keys are compared in constant time via the `subtle` crate and stored as SHA-256 hashes in SQLite.

**NCCL hangs:** `TORCH_NCCL_BLOCKING_WAIT=1` is set in `scripts/launch_vllm.sh` so NCCL errors surface instead of hanging. `scripts/watchdog.py` can supervise and restart failed processes.

**KV cache pressure:** If `vllm:gpu_cache_usage_perc` exceeds 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS` in `.env`. Alert fires at 85%.

**CUDA device order:** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (default in `launch_vllm.sh`) for consistent device indexing. Override `CUDA_VISIBLE_DEVICES` in `.env` to pin specific GPUs.

**Cold start and torch.compile:** `launch_vllm.sh` passes `--enforce-eager` which **skips torch.compile entirely**. This avoids an indefinite hang observed on this hardware (workers consumed 46 GB VRAM at 0% utilization for 7+ hours). Triton kernels are still compiled on first run and cached in `.triton_cache/` in the project root; subsequent starts take ~30 seconds. Remove `--enforce-eager` in `VLLM_EXTRA_ARGS` if you want to benchmark torch.compile on a fresh system.

**Slow tokenizer:** `--tokenizer-mode slow` is set in `launch_vllm.sh` to work around a compatibility issue between the fast tokenizer backend (`tokenizers` Rust library) and vLLM 0.16.0. Remove this flag if a future vLLM release resolves it.

**CPU swap:** `SWAP_SPACE_GB=8` configures vLLM to spill KV blocks to RAM when GPU cache is full. Use as a safety valve only — PCIe bandwidth makes active swapping very slow.

**First Rust build:** `make gateway` compiles from source on first run (~60s). Use `make build-rust` to pre-compile.

**vLLM + transformers compatibility:** vLLM 0.16.0 requires `transformers<5`. Models using architectures that only landed in transformers 5.x (e.g. `qwen3_5_moe`, `glm4_moe_lite`) need the vLLM nightly wheel. Check the model card for minimum version requirements before downloading.
