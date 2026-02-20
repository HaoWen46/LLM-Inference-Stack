from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total requests received by the gateway",
    ["method", "endpoint", "status_code", "model"],
)

REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request latency",
    ["endpoint", "model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

TTFT = Histogram(
    "gateway_time_to_first_token_seconds",
    "Time from request receipt to first token in streaming response",
    ["model"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)

TOKENS_GENERATED = Counter(
    "gateway_tokens_generated_total",
    "Cumulative output tokens generated",
    ["model"],
)

TOKENS_PROMPTED = Counter(
    "gateway_tokens_prompted_total",
    "Cumulative input/prompt tokens",
    ["model"],
)

ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Requests currently in flight",
)

UPSTREAM_ERRORS = Counter(
    "gateway_upstream_errors_total",
    "Errors returned by vLLM upstream",
    ["status_code"],
)

AUTH_FAILURES = Counter(
    "gateway_auth_failures_total",
    "Requests rejected due to bad API key",
)

RATE_LIMITED = Counter(
    "gateway_rate_limited_total",
    "Requests rejected by rate limiter",
)
