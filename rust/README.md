# rust/ — Production Rust/Axum Gateway & GPU Exporter

## Overview

| Crate | Purpose |
|-------|---------|
| **gateway** | Rust/Axum server — auth, rate limiting, quota, proxy to vLLM |
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

vLLM is **never exposed directly** — all client traffic goes through the gateway.

## Gateway (`gateway/`)

### What It Does

| Layer | Responsibility |
|-------|-----------------|
| **Authentication** | `Authorization: Bearer` or `X-API-Key`; SHA-256 hash → DashMap lookup (O(1), no DB I/O on hot path) |
| **Key Management** | PostgreSQL-backed key store with CRUD admin API; in-memory cache refreshed every 60s |
| **Rate Limiting** | Per-IP GCRA via `governor`; background task evicts idle entries after 1 hour |
| **Body Limit** | `DefaultBodyLimit::max(4 MB)` — rejects oversized payloads with 413 before auth |
| **Quota** | Per-key daily token usage; DashMap fast path + PostgreSQL persistence; flushed every 10s |
| **Circuit Breaker** | Lockless atomic state machine; failfast if vLLM is down |
| **Proxy** | HTTP/2 connection pool to vLLM; SSE passthrough with line buffering; TTFT instrumentation |
| **Security Headers** | `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection` on every response |
| **Error Sanitization** | Internal details logged only; clients receive generic messages |
| **Observability** | Prometheus metrics + structured JSON logging + OpenTelemetry OTLP to Jaeger |

### Key Files

```
gateway/src/
├── main.rs              # Router, graceful shutdown (30s drain on SIGTERM)
├── config.rs            # Read all env vars at startup
├── proxy.rs             # Core logic: reqwest → tokio::mpsc → Body stream
├── auth.rs              # ApiKey extractor; SHA-256 hash → DashMap lookup
├── keys.rs              # KeyStore: PgPool + DashMap cache; key generation & CRUD
├── admin.rs             # Admin REST API (/admin/keys)
├── rate_limiter.rs      # Per-IP GCRA using governor
├── quota.rs             # Per-key token quota; DashMap + PostgreSQL flush
├── circuit_breaker.rs   # AtomicU8 state + AtomicU64 failure counter
├── metrics.rs           # prometheus-client registry
├── error.rs             # Error types and HTTP responses
├── tracing_setup.rs     # JSON logging + optional Jaeger OTLP
migrations/
├── 001_api_keys.sql     # api_keys table
└── 002_token_usage.sql  # token_usage table (replaces SQLite)
```

### Building & Running

```bash
# Compile (first time ~60s, cached thereafter)
make build-rust
# or: cargo build --manifest-path rust/Cargo.toml --release

# Start PostgreSQL first (required)
make db

# Run the gateway (starts in foreground, migrations run automatically)
make gateway
```

### Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /v1/chat/completions` | API key | Chat completion (streaming or non-streaming) |
| `POST /v1/completions` | API key | Raw text completion |
| `GET /v1/models` | API key | List served models from vLLM |
| `GET /v1/models/local` | API key | List model weights on disk (GGUF + HuggingFace) |
| `GET /v1/usage` | API key | Per-key daily token usage and quota |
| `POST /admin/keys` | Admin key | Create a new API key (plaintext shown once) |
| `GET /admin/keys` | Admin key | List all keys with metadata |
| `PATCH /admin/keys/:id` | Admin key | Enable or disable a key |
| `DELETE /admin/keys/:id` | Admin key | Delete a key permanently |
| `GET /health` | — | Liveness check |
| `GET /ready` | — | Readiness probe; polls vLLM `/health` |
| `GET /metrics` | — | Prometheus metrics (text format) |

### API Key Management

Keys are stored in PostgreSQL (`api_keys` table) and cached in a DashMap for O(1) auth lookups.

```bash
# Create a key
curl -X POST http://localhost:8080/admin/keys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": "alice"}' | jq .
# → {"id": "...", "label": "alice", "key": "gw_...", "created_at": "..."}
# The plaintext key is shown exactly once.

# List keys
curl http://localhost:8080/admin/keys \
  -H "Authorization: Bearer $ADMIN_KEY" | jq .

# Disable a key
curl -X PATCH http://localhost:8080/admin/keys/<id> \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Delete a key
curl -X DELETE http://localhost:8080/admin/keys/<id> \
  -H "Authorization: Bearer $ADMIN_KEY"
```

**Migration from `GATEWAY_API_KEYS`**: if `GATEWAY_API_KEYS` is set and the database has no keys, the gateway inserts each key automatically on first startup (labelled `migrated-0`, `migrated-1`, …). After that, use the admin API.

### Example Requests

```bash
GATEWAY_HOST=localhost:8080
API_KEY=gw_...   # obtained from POST /admin/keys

# Chat completion (streaming)
curl -X POST http://$GATEWAY_HOST/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Check daily quota usage
curl -H "Authorization: Bearer $API_KEY" \
  http://$GATEWAY_HOST/v1/usage | jq .
```

### Configuration

Variables must be exported in the shell or set in Docker env_file — no dotenv. See `config/.env.example` for reference.

**Required:**
```bash
VLLM_API_KEY=secret          # Internal Bearer token to vLLM
DATABASE_URL=postgresql://gateway:changeme@localhost:5432/gateway
ADMIN_KEY=admin-secret       # Bearer token for /admin/* endpoints
```

**Optional** (shown with defaults):
```bash
# Upstream
VLLM_URL=http://localhost:8000
SERVED_MODEL_NAME=llama-7b

# Listen
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8080

# Seeding (one-time): if set and DB has no keys, each key is auto-inserted at startup
GATEWAY_API_KEYS=dev-key-1,dev-key-2

# Rate limiting & timeouts
RATE_LIMIT_PER_MINUTE=60
REQUEST_TIMEOUT_SECONDS=300
CONNECT_TIMEOUT_SECONDS=5

# Logging & tracing
LOG_LEVEL=INFO
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Circuit breaker
CB_FAILURE_THRESHOLD=5
CB_RECOVERY_TIMEOUT=30
CB_HALF_OPEN_MAX_CALLS=1

# Token quota (0 = unlimited)
DAILY_TOKEN_QUOTA=0

# Local model discovery
MODEL_CACHE_DIR=models
```

### Local Model Discovery

`GET /v1/models/local` scans `MODEL_CACHE_DIR` without querying vLLM. Supports:
- **GGUF**: recursively finds `.gguf` files
- **HuggingFace Hub cache**: `models--org--repo` → `org/repo`
- **Flat snapshot**: `org/repo/` with `config.json`

### Performance Notes

- Sub-millisecond proxy overhead; thousands of concurrent connections (all async)
- Auth hot path: SHA-256 (stack-allocated) → DashMap lookup — no await, no DB I/O
- HTTP/2 keep-alive pool to vLLM; reuses connections
- Streaming: `stream_options: {include_usage: true}` injected so vLLM returns exact token counts in the final SSE chunk
- Graceful shutdown: 30s drain on SIGTERM, then quota flushed to PostgreSQL

## GPU Exporter (`gpu-exporter/`)

Exposes GPU metrics (utilization, memory, temperature, power) to Prometheus, scraped every 15s.

```bash
make gpu-exporter   # starts on :9101
curl http://localhost:9101/metrics | grep gpu_
```

| Metric | Unit | Description |
|--------|------|-------------|
| `gpu_utilization_percent` | % | Compute utilization |
| `gpu_memory_used_mib` | MiB | VRAM in use |
| `gpu_memory_free_mib` | MiB | Available VRAM |
| `gpu_memory_total_mib` | MiB | Total VRAM |
| `gpu_memory_utilization_percent` | % | Memory bandwidth utilization |
| `gpu_temperature_celsius` | °C | Die temperature |
| `gpu_power_draw_watts` | W | Current power draw |
| `gpu_power_limit_watts` | W | TDP limit |
| `gpu_sm_clock_mhz` | MHz | SM clock |
| `gpu_mem_clock_mhz` | MHz | Memory clock |
| `gpu_fan_speed_percent` | % | Fan speed |
| `gpu_ecc_errors_corrected_total` | count | Corrected ECC errors |
| `gpu_ecc_errors_uncorrected_total` | count | Uncorrectable ECC errors (fatal) |

## Deployment

### Single Binary

```bash
cargo build --manifest-path rust/Cargo.toml --release
cp rust/target/release/gateway /usr/local/bin/gateway

export VLLM_API_KEY=secret
export DATABASE_URL=postgresql://gateway:changeme@localhost:5432/gateway
export ADMIN_KEY=admin-secret
/usr/local/bin/gateway
```

### Docker

```bash
docker build -f docker/Dockerfile.gateway-rust -t llm-gateway:latest .

docker run -d \
  --name gateway \
  -p 8080:8080 \
  -e VLLM_API_KEY=secret \
  -e DATABASE_URL=postgresql://gateway:changeme@postgres:5432/gateway \
  -e ADMIN_KEY=admin-secret \
  -e VLLM_URL=http://vllm:8000 \
  llm-gateway:latest
```

Use `docker/docker-compose.yml` which includes PostgreSQL, Prometheus, Grafana, Jaeger, Loki, and Promtail.

### High Availability

Multiple gateway instances can share the same PostgreSQL database — key state and quota data are consistent across instances with no additional configuration.

## Security

| Layer | Mechanism |
|-------|-----------|
| **Authentication** | SHA-256 hash → DashMap lookup; `expires_at` enforced per key |
| **Key Management** | Plaintext never stored or logged; DB stores SHA-256 hex only |
| **Admin API** | Separate `ADMIN_KEY` required for all `/admin/*` routes |
| **Rate Limiting** | Per-IP GCRA via `governor`; burst protection |
| **Body Limits** | 4 MB max; rejected before auth or parsing |
| **Security Headers** | `nosniff`, `DENY`, `XSS-Protection` on all responses |
| **Error Sanitization** | Internal details (IPs, socket errors) logged only |
| **Circuit Breaker** | Returns 503 instead of timing out when vLLM is down |

## Development

```bash
# Type-check without full compile
cargo check --manifest-path rust/Cargo.toml

# Full release build
cargo build --manifest-path rust/Cargo.toml --release

# Debug logs
RUST_LOG=debug make gateway
```

## Troubleshooting

**Gateway won't start — "DATABASE_URL is required"**
```bash
export DATABASE_URL=postgresql://gateway:changeme@localhost:5432/gateway
make db   # start postgres via Docker (requires Docker daemon)
```

`make db` requires the Docker daemon. On cluster nodes without root access, run a user-space PostgreSQL via conda instead:
```bash
conda install -y -p ~/.local/share/pgenv -c conda-forge postgresql --no-default-packages
~/.local/share/pgenv/bin/initdb -D ~/.local/share/pgdata -U gateway --no-locale --encoding=UTF8
~/.local/share/pgenv/bin/pg_ctl -D ~/.local/share/pgdata -o "-p 5432" start
~/.local/share/pgenv/bin/createdb -h localhost -p 5432 -U gateway gateway
export DATABASE_URL=postgresql://gateway@localhost:5432/gateway
```

**Gateway won't start — "VLLM_API_KEY is required"**
```bash
export VLLM_API_KEY=your-secret
```

**Requests return 401 after creating a key**
The key cache refreshes every 60 seconds. Newly created keys are added to the cache immediately; if you see stale rejections after a `PATCH` or `DELETE`, wait up to 60s or restart the gateway.

**Requests return 502 Bad Gateway**
vLLM is down or unreachable.
```bash
curl http://localhost:8000/health
curl http://localhost:8080/ready
```

**Circuit breaker stuck OPEN**
vLLM will enter HALF_OPEN after `CB_RECOVERY_TIMEOUT` (default 30s) and probe with one request. Restart vLLM or the gateway to reset immediately.

**Quota not enforcing / always -1 remaining**
`DAILY_TOKEN_QUOTA=0` means unlimited. Set a non-zero value to enforce a daily cap (resets at UTC midnight).

**Streaming requests hang**
KV cache pressure — reduce `MAX_NUM_SEQS` or `MAX_MODEL_LEN` in vLLM config, or increase `REQUEST_TIMEOUT_SECONDS`.

**`make stop` terminates itself instead of the services**
`pkill -f <pattern>` searches the full command line of every process including its own parent shell, which contains the pattern literal in its argv. `make stop` avoids this by using `pgrep -x` (match by process comm name) for Rust binaries and `pgrep 'VLLM'` for vLLM workers, plus `pgrep -f 'vllm[.]entrypoints'` (the `[.]` character class matches a dot but the literal string `vllm[.]entrypoints` does not). If you write custom stop scripts, use the same techniques to avoid self-termination.

## See Also

- [../CLAUDE.md](../CLAUDE.md) — Architecture and config reference
- [../docker/docker-compose.yml](../docker/docker-compose.yml) — Full monitoring stack
