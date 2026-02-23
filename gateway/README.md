# gateway/ — Python/FastAPI Reference Implementation

## Status: Archived (Reference Only)

This directory contains the **original Python/FastAPI implementation** of the LLM inference gateway. It has been **superseded by the Rust/Axum gateway** in the `rust/` directory for production use.

The Rust gateway provides the same features with:
- Zero garbage collection overhead
- Single compiled binary (no Python runtime)
- Lower latency and memory footprint
- Same authentication, rate limiting, quota, and circuit breaker semantics

## What Is This?

The gateway is the **client-facing API layer** that sits between clients and the vLLM inference server. It provides:

| Feature | Purpose |
|---------|---------|
| **Authentication** (`auth.py`) | Bearer token validation (`Authorization: Bearer <key>` or `X-API-Key: <key>`) |
| **Rate Limiting** | Per-IP request rate limiting using GCRA (Generic Cell Rate Algorithm) |
| **Quota Tracking** (`quota.py`, `usage_db.py`) | Per-API-key daily token usage accounting |
| **Circuit Breaker** (`circuit_breaker.py`) | Failfast when vLLM is unhealthy; graceful degradation |
| **Metrics** (`metrics.py`) | Prometheus-compatible metrics (request counts, latencies, TTFT) |
| **Proxy** | OpenAI-compatible routing to vLLM backend |

## Architecture

```
Client
  ↓
Gateway (this file) — auth, rate limit, quota check
  ↓
vLLM :8000 — inference backend (not exposed to clients)
  ↓
GPU 0/1 — computation
```

## File Structure

```
gateway/
├── app.py                 # FastAPI application root
├── server.py             # uvicorn entry point
├── config.py             # Configuration from environment
├── auth.py               # Bearer token validation
├── circuit_breaker.py    # Failover control for vLLM
├── rate_limiter.py       # Per-IP request rate limiting
├── quota.py              # Per-key token quota tracking
├── usage_db.py           # SQLite quota persistence
├── metrics.py            # Prometheus metrics
└── logging_config.py     # Structured logging setup
```

## Running (Not Recommended — Use Rust Instead)

```bash
# Install deps
make setup

# Start vLLM in another terminal
make vllm

# Run the gateway
python -m gateway.server
```

## Why Rust Now?

The Rust gateway (`rust/gateway/`) was implemented because:

1. **Performance**: ~10x faster request handling with zero GC pauses
2. **Memory**: Single-binary deployment; no Python interpreter overhead
3. **Reliability**: Type safety catches configuration errors at compile time
4. **Operational**: One artifact to deploy instead of Python + dependencies

## Key Configuration

Both gateways read the same environment variables from `config/.env`:

```bash
VLLM_URL=http://localhost:8000           # vLLM endpoint
GATEWAY_PORT=8080                        # Listen port
GATEWAY_API_KEYS=key1,key2,key3          # Client bearer tokens
VLLM_API_KEY=internal-secret             # Internal auth to vLLM
```

See `config/.env.example` for the complete list.

## Migration Path

If you want to understand how the system works, the Python code is clearer. But for production deployments, always use the Rust gateway in `rust/`.
