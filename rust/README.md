# rust/ — Production Rust/Axum Gateway & GPU Exporter

## Overview

This directory contains the **active production gateway** and monitoring exporter for the LLM inference stack:

| Crate | Purpose |
|-------|---------|
| **gateway** | Rust/Axum server replacing the Python FastAPI implementation |
| **gpu-exporter** | Prometheus exporter for GPU metrics |

Both are zero-GC, single-binary services compiled to `target/release/`.

## Architecture

```
Client → Gateway :8080 (Rust/Axum)
           ↓ (HTTP/2, keep-alive)
        vLLM :8000 (Python)
           ↓
        GPU 0, GPU 1

GPU Exporter :9101 → Prometheus :9090 → Grafana :3000
```

**Key**: The gateway handles all client-facing logic. vLLM is **never exposed directly**.

## Gateway (`gateway/`)

### What It Does

The gateway is an authentication + rate limiting + quota proxy in front of vLLM:

| Layer | Responsibility |
|-------|-----------------|
| **Authentication** | Validates `Authorization: Bearer <key>` (SHA-256 hashed) |
| **Rate Limiting** | Per-IP GCRA via `governor` crate; defaults to 1000 req/sec/IP |
| **Quota** | Per-key daily token usage; DashMap fast path + SQLite persistence |
| **Circuit Breaker** | Lockless atomic state machine; failfast if vLLM is down |
| **Proxy** | HTTP/2 connection pool to vLLM; streams SSE responses efficiently |
| **Metrics** | Prometheus histogram/counter/gauge metrics |
| **Observability** | Structured JSON logging + OpenTelemetry OTLP to Jaeger |

### Key Files

```
gateway/src/
├── main.rs              # Router, graceful shutdown (30s drain on SIGTERM)
├── config.rs            # Read all env vars at startup
├── proxy.rs             # Core logic: reqwest → tokio::mpsc → Body stream
├── auth.rs              # ApiKey extractor; SHA-256 hashing
├── rate_limiter.rs      # Per-IP GCRA using governor
├── quota.rs             # Per-key token quota; DashMap + SQLite flush
├── circuit_breaker.rs   # AtomicU8 state + AtomicU64 failure counter
├── metrics.rs           # prometheus-client registry
├── error.rs             # Error types and HTTP responses
├── tracing_setup.rs     # JSON logging + optional Jaeger OTLP
└── Cargo.toml           # Dependencies
```

### Building & Running

```bash
# Compile (first time ~60s, cached thereafter)
make build-rust
# or: cargo build --manifest-path rust/Cargo.toml --release

# Run the gateway (starts in foreground)
make gateway
# Reads from config/.env, starts on GATEWAY_PORT (default 8080)

# Run in background
cargo run --manifest-path rust/Cargo.toml -p gateway --release > gateway.log 2>&1 &
```

### OpenAI-Compatible Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /v1/chat/completions` | Required | Chat completion (streaming or non-streaming) |
| `POST /v1/completions` | Required | Raw text completion |
| `GET /v1/models` | Required | List served models |
| `GET /v1/usage` | Required | Per-key daily token usage |
| `GET /health` | — | Liveness (no auth) |
| `GET /ready` | — | Readiness; polls vLLM `/health` |
| `GET /metrics` | — | Prometheus text format |

### Example Request

```bash
# Set your gateway host and key
GATEWAY_HOST=localhost:8080
API_KEY=$(grep GATEWAY_API_KEYS config/.env | cut -d= -f2 | cut -d, -f1)

curl -X POST http://$GATEWAY_HOST/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "temperature": 0.7
  }' | jq .
```

### Configuration

All config is read from `config/.env` via `Config::from_env()` at startup. See `config/.env.example` for all options:

```bash
VLLM_URL=http://localhost:8000           # vLLM endpoint (required)
GATEWAY_PORT=8080                        # Listen port (optional, default 8080)
GATEWAY_HOST=127.0.0.1                   # Listen host (optional)
GATEWAY_API_KEYS=key1,key2,key3          # Client bearer tokens (comma-separated)
VLLM_API_KEY=internal-secret             # Internal auth to vLLM
SERVED_MODEL_NAME=...                    # Model name returned by /v1/models
ENABLE_OTLP_TRACING=false               # Send traces to Jaeger
OTLP_ENDPOINT=http://localhost:4317      # Jaeger OTLP receiver
```

### Performance Notes

- **Latency**: Sub-millisecond request handling (proxy overhead < 1ms)
- **Memory**: ~50MB resident (minimal allocations)
- **Concurrency**: All async; handles thousands of concurrent connections
- **Connections**: HTTP/2 keep-alive pool to vLLM; reuses connections
- **Graceful Shutdown**: 30-second drain period on SIGTERM; waits for in-flight requests

## GPU Exporter (`gpu-exporter/`)

### What It Does

Exposes GPU metrics (utilization, memory, temperature, power) to Prometheus.

```
GPU 0, GPU 1 (nvidia-ml-py)
    ↓
GPU Exporter :9101
    ↓
Prometheus :9090 (scrapes every 15s)
    ↓
Grafana :3000 (visualizes)
```

### Running

```bash
make gpu-exporter
# Starts on :9101

# Check metrics
curl http://localhost:9101/metrics | grep gpu_
```

### Metrics

| Metric | Type | Example |
|--------|------|---------|
| `gpu_utilization_percent` | Gauge | GPU compute load |
| `gpu_memory_used_gb` | Gauge | VRAM in use |
| `gpu_memory_free_gb` | Gauge | Available VRAM |
| `gpu_temperature_celsius` | Gauge | GPU die temperature |
| `gpu_power_watts` | Gauge | Current power draw |

## Build System

The workspace uses a single `Cargo.toml` at `rust/Cargo.toml` with two members:

```toml
[workspace]
members = ["gateway", "gpu-exporter"]
resolver = "2"
```

Both compile independently but share dependencies. Build times:
- **First build**: ~60s (Rust toolchain, dependencies)
- **Incremental**: ~5-10s (local changes only)
- **Release binary**: Single-pass compilation (no incremental)

## Deployment

### Single Binary
```bash
# Copy just the binary
cp rust/target/release/gateway /usr/local/bin/gateway
cp rust/target/release/gpu-exporter /usr/local/bin/gpu-exporter

# Run with systemd/supervisor
gateway_start
gpu-exporter_start
```

### With Docker
```dockerfile
# Build stage
FROM rust:latest AS builder
COPY rust /build/rust
WORKDIR /build/rust
RUN cargo build -p gateway --release

# Runtime
FROM debian:bookworm-slim
COPY --from=builder /build/rust/target/release/gateway /usr/local/bin/
ENV VLLM_URL=http://vllm:8000
CMD ["gateway"]
```

## Development

### Adding a Feature

1. Create a new module in `gateway/src/` (e.g., `my_feature.rs`)
2. Add `mod my_feature;` to `main.rs`
3. Use types/functions from that module in routes
4. Run tests/checks:
   ```bash
   cargo check --manifest-path rust/Cargo.toml
   cargo build --manifest-path rust/Cargo.toml --release
   ```

### Debugging

Enable RUST_LOG for structured logs:

```bash
RUST_LOG=debug cargo run --manifest-path rust/Cargo.toml -p gateway --release
```

JSON output:
```json
{"timestamp":"2025-02-23T...", "level":"INFO", "message":"Starting gateway", "port":8080}
```

## Comparison: Python vs Rust

| Aspect | Python (gateway/) | Rust (rust/gateway) |
|--------|------|--------|
| **Start time** | 2-5 seconds | <100ms |
| **Memory** | 200-300MB | 50MB |
| **GC pauses** | Yes (blocking) | No (zero-copy) |
| **Concurrency** | ~100 efficient connections | ~10,000+ connections |
| **Deployment** | Requires Python + deps | Single binary |
| **Status** | Reference only | Production (active) |

## See Also

- [../gateway/README.md](../gateway/README.md) — Python reference implementation
- [../CLAUDE.md](../CLAUDE.md) — Full architecture and config reference
- [../docker/docker-compose.yml](../docker/docker-compose.yml) — Monitoring stack
