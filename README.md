# LLM Inference Stack

Production-grade LLM serving on 2x NVIDIA RTX A6000 (48GB each).

Built on [vLLM](https://github.com/vllm-project/vllm) with a FastAPI gateway, Prometheus metrics, and Grafana dashboards.

## Architecture

```
Client
  │
  ▼
NGINX  (TLS termination, optional)
  │
  ▼
Gateway :8080  (auth, rate-limit, logging, metrics)
  │
  ▼
vLLM   :8000  (inference, KV cache, tensor parallelism)
  │
  ├── GPU 0  RTX A6000 48GB
  └── GPU 1  RTX A6000 48GB

Sidecar processes
  gpu_exporter  :9101  →  Prometheus :9090  →  Grafana :3000
```

## Hardware requirements

| Component | Spec |
|-----------|------|
| GPU | 2x NVIDIA RTX A6000 (48GB VRAM each) |
| RAM | 128GB system |
| Storage | NVMe SSD recommended for model loading |
| OS | Ubuntu 22.04 |
| CUDA | 12.x |

## Stack

| Component | Purpose |
|-----------|---------|
| vLLM 0.6.6 | Inference engine — paged attention, continuous batching, tensor parallelism |
| FastAPI + uvicorn | Async API gateway |
| slowapi | Per-IP rate limiting |
| structlog | Structured JSON logging |
| prometheus-client | Metrics exposition |
| Prometheus 2.55 | Metrics storage |
| Grafana 11.3 | Dashboards and alerting |

## Quickstart

### 1. Install dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all packages
make setup
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

# Pane 2 — API gateway
make gateway

# Pane 3 — GPU metrics exporter
make gpu-exporter
```

### 5. Start monitoring

Requires Docker for Prometheus + Grafana:

```bash
make monitoring
# Prometheus → http://localhost:9090
# Grafana    → http://localhost:3000   (admin / admin)
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
│   ├── .env               ← your local config (gitignored)
│   └── .env.example       ← template
├── gateway/
│   ├── app.py             ← FastAPI app: proxy, streaming, auth, metrics
│   ├── auth.py            ← Bearer / x-api-key validation
│   ├── config.py          ← pydantic-settings from .env
│   ├── logging_config.py  ← structlog JSON setup
│   ├── metrics.py         ← Prometheus metrics definitions
│   └── server.py          ← uvicorn entrypoint
├── scripts/
│   ├── launch_vllm.sh     ← starts vLLM with settings from .env
│   ├── download_model.sh  ← huggingface_hub snapshot_download
│   ├── gpu_exporter.py    ← nvidia-smi → Prometheus on :9101
│   └── watchdog.py        ← process supervisor with health checks
├── loadtest/
│   └── runner.py          ← async load driver, rich live output, percentile report
├── observability/
│   ├── prometheus/
│   │   ├── prometheus.yml ← scrape config (vLLM + gateway + GPU exporter)
│   │   └── alerts.yml     ← KV cache pressure, latency, ECC, queue depth
│   └── grafana/
│       ├── dashboards/    ← auto-provisioned LLM stack dashboard
│       └── provisioning/  ← datasource + dashboard provider config
├── docker/
│   ├── Dockerfile.vllm
│   ├── Dockerfile.gateway
│   └── docker-compose.yml ← gateway + Prometheus + Grafana
├── pyproject.toml
└── Makefile               ← setup / download / vllm / gateway / loadtest / status / stop
```

## Makefile targets

```
make setup          create .env and install all deps into .venv
make download       pull model weights from HuggingFace
make vllm           start vLLM server (foreground)
make gateway        start API gateway (foreground)
make gpu-exporter   start GPU Prometheus exporter
make monitoring     start Prometheus + Grafana via docker-compose
make loadtest       run load test against gateway
make status         show GPU state and process health
make stop           kill all background processes
```

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
- **Errors** — 4xx/5xx by status code, auth failures, rate-limited requests

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

## Production notes

**Do not expose vLLM directly.** It has no auth and minimal error handling. Always route through the gateway.

**NCCL hangs:** Set `TORCH_NCCL_BLOCKING_WAIT=1` (already in the launch script) so NCCL errors surface instead of hanging indefinitely. The watchdog in `scripts/watchdog.py` restarts processes that fail health checks.

**KV cache pressure:** If `vllm:gpu_cache_usage_perc` exceeds 90%, lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS` in `.env`. The alert fires at 85%.

**Cold start:** First request after startup is slow — vLLM compiles Triton kernels on first run and caches them in `.triton_cache/`. Subsequent starts are faster. The gateway's `/ready` endpoint blocks until vLLM is healthy.

**CPU swap:** `SWAP_SPACE_GB=8` in `.env` configures vLLM to spill KV blocks to RAM when GPU cache is full. Use as a safety valve only — PCIe bandwidth makes active swapping very slow.
