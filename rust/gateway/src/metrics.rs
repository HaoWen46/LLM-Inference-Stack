use prometheus_client::{
    encoding::{text::encode, EncodeLabelSet},
    metrics::{counter::Counter, family::Family, gauge::Gauge, histogram::Histogram},
    registry::Registry,
};
use std::sync::{Arc, Mutex};

// ── Label types ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabels {
    pub method: String,
    pub endpoint: String,
    pub status_code: String,
    pub model: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EndpointModelLabels {
    pub endpoint: String,
    pub model: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ModelLabel {
    pub model: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct StatusCodeLabel {
    pub status_code: String,
}

// ── Histogram bucket constants ───────────────────────────────────────────────

const LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

const TTFT_BUCKETS: &[f64] = &[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];

// ── AppMetrics ───────────────────────────────────────────────────────────────

pub struct AppMetrics {
    // Mutex-wrapped registry for thread-safe text encoding
    registry: Mutex<Registry>,

    pub request_count: Family<RequestLabels, Counter>,
    pub request_latency: Family<EndpointModelLabels, Histogram, fn() -> Histogram>,
    pub ttft: Family<ModelLabel, Histogram, fn() -> Histogram>,
    pub tokens_generated: Family<ModelLabel, Counter>,
    pub tokens_prompted: Family<ModelLabel, Counter>,
    pub active_requests: Gauge,
    pub upstream_errors: Family<StatusCodeLabel, Counter>,
    pub auth_failures: Counter,
    pub rate_limited: Counter,
    pub circuit_breaker_state: Gauge,
    pub quota_exceeded: Counter,
    pub inflight_requests: Gauge,
}

impl AppMetrics {
    pub fn new() -> Arc<Self> {
        let mut registry = Registry::default();

        let request_count = Family::<RequestLabels, Counter>::default();
        registry.register(
            "gateway_requests_total",
            "Total requests received by the gateway",
            request_count.clone(),
        );

        let request_latency: Family<EndpointModelLabels, Histogram, fn() -> Histogram> =
            Family::new_with_constructor(|| {
                Histogram::new(LATENCY_BUCKETS.iter().copied())
            });
        registry.register(
            "gateway_request_duration_seconds",
            "End-to-end request latency",
            request_latency.clone(),
        );

        let ttft: Family<ModelLabel, Histogram, fn() -> Histogram> =
            Family::new_with_constructor(|| {
                Histogram::new(TTFT_BUCKETS.iter().copied())
            });
        registry.register(
            "gateway_time_to_first_token_seconds",
            "Time from request receipt to first token in streaming response",
            ttft.clone(),
        );

        let tokens_generated = Family::<ModelLabel, Counter>::default();
        registry.register(
            "gateway_tokens_generated_total",
            "Cumulative output tokens generated",
            tokens_generated.clone(),
        );

        let tokens_prompted = Family::<ModelLabel, Counter>::default();
        registry.register(
            "gateway_tokens_prompted_total",
            "Cumulative input/prompt tokens",
            tokens_prompted.clone(),
        );

        let active_requests = Gauge::default();
        registry.register(
            "gateway_active_requests",
            "Requests currently in flight",
            active_requests.clone(),
        );

        let upstream_errors = Family::<StatusCodeLabel, Counter>::default();
        registry.register(
            "gateway_upstream_errors_total",
            "Errors returned by vLLM upstream",
            upstream_errors.clone(),
        );

        let auth_failures = Counter::default();
        registry.register(
            "gateway_auth_failures_total",
            "Requests rejected due to bad API key",
            auth_failures.clone(),
        );

        let rate_limited = Counter::default();
        registry.register(
            "gateway_rate_limited_total",
            "Requests rejected by rate limiter",
            rate_limited.clone(),
        );

        let circuit_breaker_state = Gauge::default();
        registry.register(
            "gateway_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            circuit_breaker_state.clone(),
        );

        let quota_exceeded = Counter::default();
        registry.register(
            "gateway_quota_exceeded_total",
            "Requests rejected due to daily token quota exhausted",
            quota_exceeded.clone(),
        );

        let inflight_requests = Gauge::default();
        registry.register(
            "gateway_inflight_requests",
            "Number of requests currently being processed",
            inflight_requests.clone(),
        );

        Arc::new(AppMetrics {
            registry: Mutex::new(registry),
            request_count,
            request_latency,
            ttft,
            tokens_generated,
            tokens_prompted,
            active_requests,
            upstream_errors,
            auth_failures,
            rate_limited,
            circuit_breaker_state,
            quota_exceeded,
            inflight_requests,
        })
    }

    /// Render all metrics in Prometheus text format.
    pub fn render(&self) -> String {
        let mut buf = String::new();
        let registry = self.registry.lock().unwrap();
        encode(&mut buf, &*registry).unwrap_or_default();
        buf
    }
}
