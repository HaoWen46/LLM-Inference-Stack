//! LLM Inference Gateway — Rust/Axum implementation.
//!
//! Replaces the Python/FastAPI gateway with a zero-GC, single-binary server.
//! Reads configuration exclusively from environment variables (no dotenv).
mod admin;
mod auth;
mod backends;
mod batches;
mod circuit_breaker;
mod config;
mod error;
mod keys;
mod metrics;
mod proxy;
mod quota;
mod rate_limiter;
mod tracing_setup;

use sqlx::PgPool;
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
    backends::BackendRouter,
    config::Config,
    error::GatewayError,
    keys::KeyStore,
    metrics::AppMetrics,
    quota::QuotaStore,
    rate_limiter::RateLimiterMap,
};

// ── Shared application state ─────────────────────────────────────────────────

pub struct AppState {
    pub config: Arc<Config>,
    /// Persistent HTTP/2 connection pool shared by all upstream calls.
    pub client: reqwest::Client,
    /// Multi-backend router with per-backend circuit breakers and health state.
    pub backend_router: Arc<BackendRouter>,
    pub rate_limiters: Arc<RateLimiterMap>,
    pub quota: Arc<QuotaStore>,
    pub key_store: Arc<KeyStore>,
    pub metrics: Arc<AppMetrics>,
    /// PostgreSQL pool — used by batch workers and admin handlers.
    pub db_pool: PgPool,
    /// Set to true once the primary backend has responded healthy (warmup complete).
    pub warmed_up: Arc<AtomicBool>,
    /// Set to true on SIGTERM/Ctrl-C; new proxy requests return 503.
    pub shutting_down: Arc<AtomicBool>,
}

// ── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Arc::new(Config::from_env()?);
    tracing_setup::init(&config)?;

    info!(
        version = env!("CARGO_PKG_VERSION"),
        backends = config.backends.len(),
        "gateway starting"
    );

    // ── PostgreSQL connection pool ────────────────────────────────────────────
    let pool = sqlx::PgPool::connect(&config.database_url).await?;
    info!("connected to PostgreSQL");

    // ── Key store (runs migrations, seeds from GATEWAY_API_KEYS if DB empty) ─
    let seed_keys: Vec<String> = std::env::var("GATEWAY_API_KEYS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|k| {
            let k = k.trim().to_string();
            if k.is_empty() { None } else { Some(k) }
        })
        .collect();

    let key_store = KeyStore::init(pool.clone(), seed_keys).await?;
    key_store.start_refresh_task();

    // ── Quota store (PostgreSQL) ───────────────────────────────────────────────
    let quota = QuotaStore::new(config.daily_token_quota, pool.clone()).await?;
    quota.start_flush_task();

    // ── Metrics ──────────────────────────────────────────────────────────────
    let app_metrics = AppMetrics::new();

    // ── Rate limiter ──────────────────────────────────────────────────────────
    let rate_limiters = Arc::new(RateLimiterMap::new(config.rate_limit_per_minute));
    rate_limiters.start_cleanup_task();

    // ── reqwest client — single shared connection pool ────────────────────────
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .pool_max_idle_per_host(100)
        .pool_idle_timeout(Duration::from_secs(30))
        .use_rustls_tls()
        .build()?;

    // ── Backend router (per-backend CB + health checks) ───────────────────────
    let backend_router = BackendRouter::new(
        config.backends.clone(),
        config.cb_failure_threshold,
        config.cb_recovery_timeout_secs,
        config.cb_half_open_max_calls,
    );
    backend_router.start_health_task(client.clone());

    for b in &backend_router.backends {
        info!(
            url = %b.url,
            models = ?b.models,
            weight = b.weight,
            "backend registered"
        );
    }

    let warmed_up = Arc::new(AtomicBool::new(false));
    let shutting_down = Arc::new(AtomicBool::new(false));

    let state = Arc::new(AppState {
        config: config.clone(),
        client: client.clone(),
        backend_router,
        rate_limiters,
        quota,
        key_store,
        metrics: app_metrics.clone(),
        db_pool: pool.clone(),
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

    let middleware = ServiceBuilder::new()
        .set_x_request_id(MakeRequestUuid)
        .propagate_x_request_id()
        .layer(TraceLayer::new_for_http())
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
        // ── Admin (ADMIN_KEY auth) ────────────────────────────────────────────
        .nest("/admin", admin::router())
        // ── Authenticated ─────────────────────────────────────────────────────
        .route("/v1/usage", get(proxy::usage_handler))
        .route("/v1/models/local", get(proxy::local_models_handler))
        .route("/v1/models", get(proxy::models_handler))
        .route("/v1/chat/completions", post(proxy::chat_completions_handler))
        .route("/v1/completions", post(proxy::completions_handler))
        .route("/v1/embeddings", post(proxy::embeddings_handler))
        .route("/v1/tokenize", post(proxy::tokenize_handler))
        .route("/v1/detokenize", post(proxy::detokenize_handler))
        .nest("/v1/batches", batches::router())
        // ── Global middleware ─────────────────────────────────────────────────
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
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
            "warming up — primary backend not yet healthy".into(),
        ));
    }

    let backend = state
        .backend_router
        .primary()
        .ok_or_else(|| GatewayError::UpstreamError("no backends configured".into()))?;

    let resp = state
        .client
        .get(format!("{}/health", backend.url))
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    if resp.status().is_success() {
        Ok((StatusCode::OK, Json(serde_json::json!({"status": "ready"}))))
    } else {
        Err(GatewayError::UpstreamError("primary backend unhealthy".into()))
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
    let backend = match state.backend_router.primary() {
        Some(b) => b,
        None => {
            tracing::warn!("no backends configured — skipping warmup");
            state.warmed_up.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    };

    let url = format!("{}/health", backend.url);
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
                info!(attempts = attempt + 1, url = %url, "gateway ready — primary backend healthy");
                return;
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
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
