# observability/

Monitoring stack config for the LLM inference stack. Start everything with:

```bash
make monitoring   # Prometheus :9090, Grafana :3000, Jaeger :16686, Loki :3100
make monitoring-down
```

## Directory layout

```
observability/
├── prometheus/
│   ├── prometheus.yml      # scrape targets + alert rule file reference
│   └── alerts.yml          # alert rules (KV cache, latency, GPU health, gateway)
├── grafana/
│   ├── dashboards/
│   │   └── llm_stack.json  # auto-provisioned dashboard
│   └── provisioning/
│       ├── dashboards/     # dashboard provider config (points at dashboards/)
│       └── datasources/    # Prometheus + Loki datasource definitions
├── loki/
│   └── loki-config.yml     # log storage config (filesystem backend)
└── promtail/
    └── promtail-config.yml # log scrape config (reads .cache/*.log)
```

## Prometheus

`prometheus.yml` scrapes three targets every 5s (GPU metrics every 2s):

| Job | Target | What it scrapes |
|-----|--------|-----------------|
| `vllm` | `host.docker.internal:8000` | vLLM built-in metrics |
| `gateway` | `host.docker.internal:8080` | Rust gateway metrics |
| `gpu` | `host.docker.internal:9101` | GPU exporter metrics |

`host.docker.internal` resolves to the Docker host — the services run on the host, not inside containers.

**To add Alertmanager** (email/Slack notifications), edit the `alerting.alertmanagers` block in `prometheus.yml`:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["localhost:9093"]
```

## Alert rules (`alerts.yml`)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `KVCachePressureWarning` | KV cache > 85% for 2m | warning |
| `KVCachePressureCritical` | KV cache > 95% for 30s | critical |
| `RequestsBeingSwapped` | KV blocks spilling to CPU | warning |
| `HighP99Latency` | P99 e2e latency > 60s for 5m | critical |
| `HighTTFTP95` | P95 TTFT > 10s for 5m | warning |
| `HighQueueDepth` | Waiting requests > 50 for 5m | warning |
| `GPUMemoryLow` | Free VRAM < 2 GB for 1m | critical |
| `GPUHighTemperature` | GPU temp > 85°C for 2m | warning |
| `GPUUncorrectedECC` | Any uncorrected ECC errors in 10m | critical |
| `GPUPowerCapHit` | Power draw > 98% of TDP for 5m | warning |
| `GatewayHighErrorRate` | 5xx rate > 5% for 5m | critical |
| `CircuitBreakerOpen` | Circuit breaker OPEN for > 2m | critical |

**To tune a threshold**, edit the `expr:` field in `alerts.yml` and reload Prometheus (`curl -X POST http://localhost:9090/-/reload`).

## Grafana dashboard

`grafana/dashboards/llm_stack.json` is auto-provisioned on first start. It covers:

- Traffic: req/s, active requests, queue depth
- Latency: P50/P95/P99 end-to-end + TTFT percentiles
- GPU: utilization, VRAM used/free, power draw, temperature, SM clock
- KV cache: usage %, CPU swap, running/waiting/swapped sequences
- Errors: 4xx/5xx by code, auth failures, rate-limited, quota exceeded

**To edit the dashboard:** make changes in the Grafana UI, then export JSON (`Dashboard settings → JSON Model → Copy to clipboard`) and overwrite `llm_stack.json`.

## Loki + Promtail

Promtail scrapes `*.log` files from `.cache/` in the project root and ships them to Loki. View logs in Grafana (`Explore → Loki`) or:

```bash
make logs   # tail Loki + Promtail container logs
```
