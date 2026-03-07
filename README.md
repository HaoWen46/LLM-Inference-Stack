# LLM Inference Stack

Production-grade LLM serving on 3× NVIDIA RTX A6000 (144 GB total VRAM).

Built on [vLLM](https://github.com/vllm-project/vllm) with a **Rust/Axum gateway**, Prometheus metrics, and Grafana dashboards.

**Current deployment:** `Qwen/Qwen3.5-27B` (BF16, reasoning model) on GPUs 0 & 1 (A6000, TP=2).

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
vLLM   :8000  (inference, KV cache, tensor parallelism — Qwen3.5-27B, TP=2)
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
| vLLM 0.17.0 nightly | Python | Inference engine — paged attention, continuous batching, tensor parallelism |
| **Axum 0.7** | **Rust** | **API gateway — zero-GC, predictable tail latency** |
| **governor** | **Rust** | **Per-IP and per-key GCRA rate limiting** |
| **sqlx + DashMap** | **Rust** | **Key store, per-key quota and limits (PostgreSQL backend)** |
| **prometheus-client** | **Rust** | **Metrics exposition** |
| Prometheus 2.55 | — | Metrics storage |
| Grafana 11.3 | — | Dashboards and alerting |
| Jaeger | — | Distributed tracing (OpenTelemetry OTLP) |
| Loki + Promtail | — | Log aggregation |

The Python FastAPI gateway (`gateway/`) is kept as a reference implementation and fallback.

## Hardware

| # | GPU | VRAM | Notes |
|---|-----|------|-------|
| 0 | NVIDIA RTX A6000 | 48 GB | **active — Qwen3.5-27B shard 0** |
| 1 | NVIDIA RTX A6000 | 48 GB | **active — Qwen3.5-27B shard 1** |
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
# Use the full local path for MODEL_NAME — avoids HuggingFace re-downloading
# weights that are already on disk when you restart vLLM.
MODEL_NAME=/home5/B11902156/models/Qwen/Qwen3.5-27B   # full local path avoids re-download
SERVED_MODEL_NAME=Qwen/Qwen3.5-27B
MODEL_CACHE_DIR=/home5/B11902156/models   # scanned by GET /v1/models/local

VLLM_PORT=8000
VLLM_API_KEY=change-me                  # internal secret between gateway and vLLM

GATEWAY_PORT=8080
DATABASE_URL=postgresql://gateway:changeme@localhost:5432/gateway
ADMIN_KEY=admin-secret-change-me        # bearer token for /admin/* key management

# Optional: if set and the DB has no keys, each entry is auto-inserted at startup.
# After first run, manage keys via POST /admin/keys instead.
GATEWAY_API_KEYS=key1,key2

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
bash scripts/download_model.sh Qwen/Qwen3.5-27B

# single GGUF file (bypasses huggingface_hub / hf_xet):
bash scripts/download_model.sh TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q8_0.gguf
```

Models are saved to `MODEL_CACHE_DIR/<repo-id>/` (configurable via `.env`).

> **Model compatibility:** Newer architectures like `qwen3_5_moe` (Qwen3.5) require
> vLLM ≥ 0.17.0. This stack runs the nightly wheel to support them. Stable
> architectures (Qwen3, Llama, Mistral, DeepSeek-R1-Distill) work with the stable
> 0.16 release.
>
### 4. Run the stack

Open two terminal panes (or use tmux):

```bash
# PostgreSQL (required for gateway key store and quota)
# Requires the Docker daemon to be running. On cluster nodes without root:
#   conda install -y -p ~/.local/share/pgenv -c conda-forge postgresql
#   initdb -D ~/.local/share/pgdata -U gateway --no-locale --encoding=UTF8
#   pg_ctl -D ~/.local/share/pgdata -o "-p 5432" start
#   createdb -h localhost -p 5432 -U gateway gateway
#   # then set: DATABASE_URL=postgresql://gateway@localhost:5432/gateway
make db

# Pane 1 — inference engine
make vllm
# or directly:
bash scripts/launch_vllm.sh

# Pane 2 — API gateway (Rust; builds on first run ~60s, migrations run automatically)
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

# Create an API key (only needed once; key is shown once in the response)
curl -X POST http://localhost:8080/admin/keys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": "dev"}' | jq .
# copy the "key" field → API_KEY=gw_...

# Create a key with per-key limits (all optional)
curl -X POST http://localhost:8080/admin/keys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "label": "alice",
    "rpm_limit": 30,
    "daily_token_limit": 500000,
    "allowed_models": ["Qwen/Qwen3.5-27B"],
    "expires_at": "2026-12-31T23:59:59Z"
  }' | jq .

# Non-streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "What is tensor parallelism?"}],
    "max_tokens": 256
  }'

# Streaming chat
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Explain MoE briefly."}],
    "max_tokens": 256,
    "stream": true
  }'

# Per-key token usage
curl http://localhost:8080/v1/usage \
  -H "Authorization: Bearer $API_KEY"
```

### 7. Load test

```bash
# API_KEY must be a valid key — either pass it explicitly or set GATEWAY_API_KEYS in .env
API_KEY=gw_... make loadtest
# tune concurrency and duration:
API_KEY=gw_... CONCURRENCY=32 DURATION=120 make loadtest
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
│   │   ├── migrations/
│   │   │   ├── 001_api_keys.sql          ← api_keys table
│   │   │   ├── 002_token_usage.sql       ← token_usage table
│   │   │   ├── 003_batches.sql           ← batches table
│   │   │   └── 004_per_key_controls.sql  ← orgs table; rpm_limit, daily_token_limit, allowed_models, org_id
│   │   └── src/
│   │       ├── main.rs       ← router, warmup, graceful shutdown
│   │       ├── config.rs     ← Config::from_env()
│   │       ├── auth.rs       ← ApiKey extractor; SHA-256 hash → DashMap lookup; expires_at check
│   │       ├── keys.rs       ← KeyStore: PgPool + DashMap cache, key generation, org CRUD
│   │       ├── admin.rs      ← /admin/keys CRUD + PUT limits; /admin/orgs CRUD + usage
│   │       ├── batches.rs    ← batch inference API; background tokio worker
│   │       ├── rate_limiter.rs ← per-IP and per-key GCRA (governor), idle eviction
│   │       ├── circuit_breaker.rs ← lockless atomics (no Mutex)
│   │       ├── quota.rs      ← DashMap cache + PostgreSQL flush every 10s; per-key limit override
│   │       ├── metrics.rs    ← prometheus-client Registry
│   │       ├── proxy.rs      ← sync + SSE streaming proxy, TTFT; model allowlist enforcement
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
│   ├── launch_vllm.sh        ← starts vLLM with settings from .env (activates .venv directly)
│   ├── download_model.sh     ← file-by-file HF download with optional staging dir; or single-file wget for GGUF
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
├── examples/
│   ├── client.py             ← interactive CLI client; key from .env or --key flag
│   ├── test_gateway.py       ← end-to-end smoke tests (auth, streaming, reasoning mode)
│   └── test_api.py           ← API integration tests (70 checks, no inference needed for most)
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
make db             start PostgreSQL (required by gateway)
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
| Authentication | `Authorization: Bearer` or `X-API-Key`; SHA-256 hash → DashMap lookup (O(1), no DB I/O on hot path); `expires_at` checked inline |
| Key management | PostgreSQL `api_keys` table; CRUD + per-key limits via `/admin/keys`; cache refreshed every 60s; plaintext never stored |
| Organisations | `orgs` table; `org_id` FK on `api_keys`; `GET /admin/orgs/:id/usage` aggregates token usage across all org keys |
| Rate limiting | **Per-IP**: GCRA via `governor`, lazy `DashMap<IpAddr, Arc<RateLimiter>>`; **per-key**: separate `DashMap<key_hash, (limiter, rpm)>`, recreated if `rpm_limit` changes; both cleaned up after 1h idle |
| Model allowlist | `allowed_models TEXT[]` on each key; checked in `proxy_request` before forwarding; empty = all models allowed; returns 403 |
| Per-key quota | `daily_token_limit` on each key overrides global `DAILY_TOKEN_QUOTA`; `0`/NULL = unlimited; `/v1/usage` shows the effective limit |
| Circuit breaker | Lockless `AtomicU8` state + `AtomicU64` failure counter; `compare_exchange` transitions |
| Quota | `DashMap` in-memory cache (atomic fast path) flushed to PostgreSQL every 10s; SHA-256 key hashing |
| Batch API | `POST /v1/batches` accepts inline `requests` array; background `tokio::spawn` worker processes items sequentially; cancellable; results persisted in `batches` table |
| Request size limit | `DefaultBodyLimit::max(4 MB)` — oversized bodies rejected with 413 before parsing or auth |
| Streaming | `reqwest::bytes_stream()` → line buffer → `tokio::mpsc` → `Body::from_stream`; injects `stream_options: {include_usage: true}` for exact token counts |
| TTFT | Measured on first non-empty `data: ` SSE line; recorded as Prometheus histogram |
| Token counting | Streaming: exact `prompt_tokens`/`completion_tokens` from vLLM's final usage chunk; sync: from `usage` field in response JSON |
| LoRA routing | `LORA_MODULES="alias=path"` parsed at startup; model name rewrite skipped for known aliases so vLLM activates the correct adapter |
| Security headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection` on every response via `SetResponseHeaderLayer` |
| Error sanitization | `UpstreamError` messages (IPs, socket details) logged internally; clients receive only `"Upstream service unavailable"` |
| Graceful shutdown | `SIGTERM`/`Ctrl-C` → `shutting_down` flag → 30s drain |
| Observability | `tracing-subscriber` JSON + optional OpenTelemetry OTLP to Jaeger |

## Memory budget

Reference for RTX A6000 (48 GB VRAM each):

| Model | Precision | Weights/GPU (TP=2) | KV headroom | Notes |
|-------|-----------|-------------------|-------------|-------|
| Qwen3.5-27B | bf16 | ~27 GB | ~19 GB | **current; 2× A6000** |
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
    api_key="gw_...",   # key from POST /admin/keys
)

# Streaming (Qwen3.5-27B is a reasoning model — thinking tokens arrive first)
stream = client.chat.completions.create(
    model="Qwen/Qwen3.5-27B",
    messages=[{"role": "user", "content": "Explain paged attention."}],
    max_tokens=512,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Authentication accepts both `Authorization: Bearer <key>` and `X-API-Key: <key>` headers.

**Qwen3.5 thinking mode:** by default the model reasons before answering (thinking tokens are separated into the `reasoning` field by vLLM's `--reasoning-parser qwen3`; the final answer is in `content`). To skip thinking and get an instant answer, pass:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

or in raw JSON: `"chat_template_kwargs": {"enable_thinking": false}` at the top level of the request body.

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
| POST | `/v1/embeddings` | ✓ | Embeddings (proxied as-is; requires a dedicated embedding model) |
| POST | `/v1/tokenize` | ✓ | Tokenize — passthrough to vLLM `/tokenize` |
| POST | `/v1/detokenize` | ✓ | Detokenize — passthrough to vLLM `/detokenize` |
| GET | `/v1/usage` | ✓ | Per-key daily token usage (shows effective quota: per-key or global) |
| POST | `/v1/batches` | ✓ | Create offline batch job (inline requests array) |
| GET | `/v1/batches` | ✓ | List batch jobs for this key |
| GET | `/v1/batches/:id` | ✓ | Get batch status |
| POST | `/v1/batches/:id/cancel` | ✓ | Cancel a running batch |
| GET | `/v1/batches/:id/results` | ✓ | Fetch completed batch results |
| POST | `/admin/keys` | Admin key | Create a key (plaintext shown once); accepts optional limits |
| GET | `/admin/keys` | Admin key | List all keys with metadata and per-key limits |
| PATCH | `/admin/keys/:id` | Admin key | Enable or disable a key |
| PUT | `/admin/keys/:id/limits` | Admin key | Replace per-key limits (rpm, daily tokens, model allowlist, expiry, org) |
| DELETE | `/admin/keys/:id` | Admin key | Delete a key permanently |
| POST | `/admin/orgs` | Admin key | Create an organisation |
| GET | `/admin/orgs` | Admin key | List all organisations |
| GET | `/admin/orgs/:id/usage` | Admin key | Aggregate token usage for all keys in an org (today) |

### `/v1/models/local` — discover models on disk

Returns every GGUF file and HuggingFace model directory found under `MODEL_CACHE_DIR`, formatted so you can copy-paste the `id` straight into your request body.

```bash
curl http://localhost:8080/v1/models/local \
  -H "Authorization: Bearer $API_KEY"
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3.5-27B",
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
| 403 | `invalid_request_error` (model not in allowlist) |
| 404 | `invalid_request_error` |
| 413 | `invalid_request_error` |
| 429 | `rate_limit_error` (rate limit or quota exceeded) |
| 5xx | `api_error` |

## Testing

### Unit tests (Python gateway, no GPU required)

```bash
# Install dev deps only (skips vllm/torch)
uv sync --only-group dev --no-install-project

# Run all 17 tests
uv run pytest tests/ -v
```

Tests cover: `/health`, `/v1/models`, `/v1/models/local`, chat completions (sync + streaming), authentication errors, invalid JSON, upstream 5xx passthrough, OpenAI error schema validation, and per-key usage endpoint. vLLM is mocked with `respx`; the usage DB is patched with `AsyncMock` stubs so tests are fully self-contained.

### Integration tests (live gateway + vLLM)

```bash
# API surface test — 70 checks covering auth, admin lifecycle, error shapes,
# security headers, quota, body-size limits, metrics (no inference for most checks)
.venv/bin/python3 examples/test_api.py

# End-to-end smoke test — system prompts, streaming, reasoning mode, /v1/completions
.venv/bin/python3 examples/test_gateway.py
```

`test_api.py` runs in a few seconds; `test_gateway.py` generates actual completions (~2 min).

### Interactive client

```bash
# Prompt from CLI (key auto-resolved from config/.env)
.venv/bin/python3 examples/client.py "What is tensor parallelism?"
.venv/bin/python3 examples/client.py --stream "Explain MoE routing."
.venv/bin/python3 examples/client.py --url http://localhost:8080 --key gw_... "Hello"
```

## Production notes

**Do not expose vLLM directly.** It has no auth and minimal error handling. Always route through the gateway.

**API key security:** create keys via `POST /admin/keys` (requires `ADMIN_KEY`). Keys are generated as `gw_<32-random-bytes-base64url>` and stored as SHA-256 hashes in PostgreSQL — plaintext is shown exactly once and never persisted. Set `ADMIN_KEY` to a strong random secret (`openssl rand -hex 32`).

**NCCL hangs:** `TORCH_NCCL_BLOCKING_WAIT=1` is set in `scripts/launch_vllm.sh` so NCCL errors surface instead of hanging. `scripts/watchdog.py` can supervise and restart failed processes.

**KV cache pressure:** If `vllm:gpu_cache_usage_perc` exceeds 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS` in `.env`. Alert fires at 85%.

**CUDA device order:** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (default in `launch_vllm.sh`) for consistent device indexing. Override `CUDA_VISIBLE_DEVICES` in `.env` to pin specific GPUs.

**Cold start and torch.compile:** `launch_vllm.sh` passes `--enforce-eager` which **skips torch.compile entirely**. This avoids an indefinite hang observed on this hardware (workers consumed 46 GB VRAM at 0% utilization for 7+ hours). Triton kernels are still compiled on first run and cached in `.triton_cache/` in the project root; subsequent starts take ~30 seconds. Remove `--enforce-eager` in `VLLM_EXTRA_ARGS` if you want to benchmark torch.compile on a fresh system.

**Slow tokenizer:** `--tokenizer-mode slow` is set in `launch_vllm.sh` because the fast tokenizer Rust library dropped support for `all_special_tokens_extended`, which the Qwen3.5 tokenizer needs.

**BF16 GEMM on torch 2.10.0 + CUDA 12.9:** torch 2.10.0 bundles `nvidia-cublas-cu12==12.8.4.1`, which has a broken BF16 GEMM on Ampere GPUs when the CUDA driver is 12.9. Fix with `uv pip install "nvidia-cublas-cu12==12.9.1.4" --no-deps`. This is a one-time fix that persists because `launch_vllm.sh` activates the venv directly (not via `uv run`).

**CPU swap:** `SWAP_SPACE_GB=8` configures vLLM to spill KV blocks to RAM when GPU cache is full. Use as a safety valve only — PCIe bandwidth makes active swapping very slow.

**First Rust build:** `make gateway` compiles from source on first run (~60s). Use `make build-rust` to pre-compile.

**vLLM nightly:** Qwen3.5 (`qwen3_5_moe` architecture) requires vLLM ≥ 0.17.0. The nightly wheel is pinned in `pyproject.toml`. Note that the nightly server only retains the most recent build — if you recreate the venv from scratch, `uv sync` may fail because the exact pinned build has rotated off. In that case, update the version in `pyproject.toml` to whichever build is currently on `wheels.vllm.ai/nightly`.
