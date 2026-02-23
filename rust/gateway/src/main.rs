//! LLM Inference Gateway — Rust/Axum implementation.
//!
//! Replaces the Python/FastAPI gateway with a zero-GC, single-binary server.
//! Reads configuration exclusively from environment variables (no dotenv).
mod auth;
mod circuit_breaker;
mod config;
mod error;
mod metrics;
mod proxy;
mod quota;
mod rate_limiter;
mod tracing_setup;

use std::{
    net::SocketAddr,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use axum::{
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};

use tokio::signal;
use tower_http::{request_id::MakeRequestUuid, trace::TraceLayer, ServiceBuilderExt};
use tracing::info;

use crate::{
    circuit_breaker::CircuitBreaker,
    config::Config,
    error::GatewayError,
    metrics::AppMetrics,
    quota::QuotaStore,
    rate_limiter::RateLimiterMap,
};

// ── Shared application state ─────────────────────────────────────────────────

pub struct AppState {
    pub config: Arc<Config>,
    /// Persistent HTTP/2 connection pool to vLLM.
    pub client: reqwest::Client,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub rate_limiters: Arc<RateLimiterMap>,
    pub quota: Arc<QuotaStore>,
    pub metrics: Arc<AppMetrics>,
    /// Set to true once vLLM has responded healthy (warmup complete).
    pub warmed_up: Arc<AtomicBool>,
    /// Set to true on SIGTERM/Ctrl-C; new proxy requests return 503.
    pub shutting_down: Arc<AtomicBool>,
}

// ── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Arc::new(Config::from_env()?);
    tracing_setup::init(&config)?;

    info!(version = env!("CARGO_PKG_VERSION"), "gateway starting");

    // Warn if operator forgot to replace the example placeholder keys.
    const DEFAULT_KEYS: &[&str] = &["dev-key-1", "dev-key-2"];
    for k in DEFAULT_KEYS {
        if config.gateway_api_keys.contains(*k) {
            tracing::warn!(
                key = k,
                "SECURITY: GATEWAY_API_KEYS contains a default placeholder — \
                 replace with a strong random secret before exposing to the internet"
            );
        }
    }

    // ── Metrics ──────────────────────────────────────────────────────────────
    let app_metrics = AppMetrics::new();

    // ── Circuit breaker ───────────────────────────────────────────────────────
    let circuit_breaker = Arc::new(CircuitBreaker::new(
        config.cb_failure_threshold,
        config.cb_recovery_timeout_secs,
        config.cb_half_open_max_calls,
    ));

    // ── Rate limiter ──────────────────────────────────────────────────────────
    let rate_limiters = Arc::new(RateLimiterMap::new(config.rate_limit_per_minute));
    rate_limiters.start_cleanup_task();

    // ── Quota store (SQLite) ──────────────────────────────────────────────────
    let quota = QuotaStore::new(config.daily_token_quota, "data/usage.db").await?;
    quota.start_flush_task();

    // ── reqwest client — single shared connection pool ────────────────────────
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .pool_max_idle_per_host(100)
        .pool_idle_timeout(Duration::from_secs(30))
        .use_rustls_tls()
        .build()?;

    let warmed_up = Arc::new(AtomicBool::new(false));
    let shutting_down = Arc::new(AtomicBool::new(false));

    let state = Arc::new(AppState {
        config: config.clone(),
        client: client.clone(),
        circuit_breaker: circuit_breaker.clone(),
        rate_limiters,
        quota,
        metrics: app_metrics.clone(),
        warmed_up: warmed_up.clone(),
        shutting_down: shutting_down.clone(),
    });

    // ── Warmup task ───────────────────────────────────────────────────────────
    tokio::spawn({
        let state = state.clone();
        async move { warmup(state).await }
    });

    // ── Router ────────────────────────────────────────────────────────────────
    let app = build_router(state.clone());

    // ── Listen ────────────────────────────────────────────────────────────────
    let addr: SocketAddr = format!("{}:{}", config.gateway_host, config.gateway_port)
        .parse()
        .expect("invalid GATEWAY_HOST:GATEWAY_PORT");

    info!(%addr, "gateway listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal(shutting_down.clone()))
    .await?;

    info!("gateway stopped");
    Ok(())
}

// ── Router construction ───────────────────────────────────────────────────────

fn build_router(state: Arc<AppState>) -> Router {
    use axum::extract::DefaultBodyLimit;
    use tower::ServiceBuilder;
    use tower_http::set_header::SetResponseHeaderLayer;

    // All global middleware in one ServiceBuilder so the response-header layers
    // run in the same proven chain as TraceLayer/RequestId (which already work).
    let middleware = ServiceBuilder::new()
        .set_x_request_id(MakeRequestUuid)
        .propagate_x_request_id()
        .layer(TraceLayer::new_for_http())
        // Security headers — added to every response after tracing is recorded.
        .layer(SetResponseHeaderLayer::overriding(
            header::HeaderName::from_static("x-content-type-options"),
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            header::HeaderName::from_static("x-frame-options"),
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            header::HeaderName::from_static("x-xss-protection"),
            HeaderValue::from_static("1; mode=block"),
        ));

    Router::new()
        // ── Public (no auth) ─────────────────────────────────────────────────
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        // ── Authenticated ─────────────────────────────────────────────────────
        .route("/v1/usage", get(proxy::usage_handler))
        .route("/v1/models", get(proxy::models_handler))
        .route("/v1/chat/completions", post(proxy::chat_completions_handler))
        .route("/v1/completions", post(proxy::completions_handler))
        // ── Global middleware ─────────────────────────────────────────────────
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024)) // 4 MB — rejects oversized bodies
        .layer(middleware)
        .with_state(state)
}

// ── Liveness / readiness handlers ────────────────────────────────────────────

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn ready_handler(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, GatewayError> {
    if !state.warmed_up.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::BadRequest(
            "warming up — vLLM not yet healthy".into(),
        ));
    }

    let resp = state
        .client
        .get(format!("{}/health", state.config.vllm_url))
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    if resp.status().is_success() {
        Ok((StatusCode::OK, Json(serde_json::json!({"status": "ready"}))))
    } else {
        Err(GatewayError::UpstreamError("vLLM upstream unhealthy".into()))
    }
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

// ── Warmup task ───────────────────────────────────────────────────────────────

async fn warmup(state: Arc<AppState>) {
    let url = format!("{}/health", state.config.vllm_url);
    for attempt in 0..60u32 {
        match state
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                state
                    .warmed_up
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                info!(attempts = attempt + 1, "gateway ready — vLLM healthy");
                return;
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    // Unblock regardless — let requests fail naturally
    state
        .warmed_up
        .store(true, std::sync::atomic::Ordering::Relaxed);
    tracing::warn!("warmup timed out after 5 minutes — marking ready anyway");
}

// ── Graceful shutdown ─────────────────────────────────────────────────────────

async fn shutdown_signal(shutting_down: Arc<AtomicBool>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let sigterm = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let sigterm = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c  => info!("received Ctrl+C"),
        _ = sigterm => info!("received SIGTERM"),
    }

    info!("shutting down — draining in-flight requests (up to 30s)");
    shutting_down.store(true, std::sync::atomic::Ordering::SeqCst);

    // Flush quota store before exit (best effort)
    tokio::time::sleep(Duration::from_secs(30)).await;
}
