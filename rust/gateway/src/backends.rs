/// Multi-backend routing: per-backend circuit breaker, health tracking, weighted selection.
///
/// `BackendRouter` holds a list of `Backend` instances parsed from the `BACKENDS`
/// environment variable (or a single legacy backend from `VLLM_URL`/`VLLM_API_KEY`).
///
/// Routing algorithm:
///   1. Filter backends to those that are available (healthy + CB not OPEN) and
///      serve the requested model name.
///   2. Weighted random selection among candidates.
///   3. If no model-specific backend matches, catch-all backends (empty models list)
///      are also considered in step 1.
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use rand::Rng;
use tracing::{info, warn};

use crate::circuit_breaker::CircuitBreaker;

// ── BackendDef — parsed from env at startup ───────────────────────────────────

/// Configuration for one upstream backend, stored in `Config`.
#[derive(Debug, Clone)]
pub struct BackendDef {
    pub url: String,
    pub api_key: String,
    /// Client-facing model names this backend serves. Empty = catch-all.
    pub models: Vec<String>,
    pub weight: u32,
    /// The model name the backend expects in the `model` request field.
    /// `None` = forward the client's model name unchanged.
    pub served_model_name: Option<String>,
}

// ── Backend — live instance ───────────────────────────────────────────────────

pub struct Backend {
    pub url: String,
    pub api_key: String,
    pub models: Vec<String>,
    pub weight: u32,
    pub served_model_name: Option<String>,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub healthy: AtomicBool,
}

impl Backend {
    fn new(
        def: BackendDef,
        cb_failure_threshold: u64,
        cb_recovery_secs: u64,
        cb_half_open: u64,
    ) -> Self {
        Self {
            url: def.url,
            api_key: def.api_key,
            models: def.models,
            weight: def.weight,
            served_model_name: def.served_model_name,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                cb_failure_threshold,
                cb_recovery_secs,
                cb_half_open,
            )),
            healthy: AtomicBool::new(true),
        }
    }

    /// True when this backend should receive traffic (healthy + CB not OPEN).
    pub fn is_available(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
            && self.circuit_breaker.state() != crate::circuit_breaker::OPEN
    }

    /// True if this backend serves `model` or is a catch-all (empty models list).
    pub fn serves_model(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|m| m == model)
    }
}

// ── BackendRouter ─────────────────────────────────────────────────────────────

pub struct BackendRouter {
    pub backends: Vec<Arc<Backend>>,
}

impl BackendRouter {
    pub fn new(
        defs: Vec<BackendDef>,
        cb_failure_threshold: u64,
        cb_recovery_secs: u64,
        cb_half_open: u64,
    ) -> Arc<Self> {
        let backends = defs
            .into_iter()
            .map(|d| Arc::new(Backend::new(d, cb_failure_threshold, cb_recovery_secs, cb_half_open)))
            .collect();
        Arc::new(Self { backends })
    }

    /// Select an available backend for `model` using weighted random selection.
    /// Returns `None` if no healthy backend can serve the request.
    pub fn select(&self, model: &str) -> Option<Arc<Backend>> {
        let candidates: Vec<&Arc<Backend>> = self
            .backends
            .iter()
            .filter(|b| b.is_available() && b.serves_model(model))
            .collect();
        pick_weighted(&candidates)
    }

    /// All backends currently marked healthy — used to fan out `/v1/models`.
    pub fn healthy_backends(&self) -> Vec<Arc<Backend>> {
        self.backends
            .iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .map(Arc::clone)
            .collect()
    }

    /// Primary (first) backend — used for tokenize/detokenize passthroughs.
    pub fn primary(&self) -> Option<Arc<Backend>> {
        self.backends.first().map(Arc::clone)
    }

    /// Spawn a background task that polls `/health` for every backend every 10 seconds,
    /// updating `Backend::healthy` accordingly.
    pub fn start_health_task(self: &Arc<Self>, client: reqwest::Client) {
        let router = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                for backend in &router.backends {
                    let url = format!("{}/health", backend.url);
                    let was_healthy = backend.healthy.load(Ordering::Relaxed);
                    let now_healthy = client
                        .get(&url)
                        .timeout(Duration::from_secs(3))
                        .send()
                        .await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false);
                    backend.healthy.store(now_healthy, Ordering::Relaxed);
                    if was_healthy && !now_healthy {
                        warn!(url = %backend.url, "backend became unhealthy");
                    } else if !was_healthy && now_healthy {
                        info!(url = %backend.url, "backend recovered");
                    }
                }
            }
        });
    }
}

// ── Weighted random selection ─────────────────────────────────────────────────

fn pick_weighted<'a>(candidates: &[&'a Arc<Backend>]) -> Option<Arc<Backend>> {
    match candidates.len() {
        0 => None,
        1 => Some(Arc::clone(candidates[0])),
        _ => {
            let total: u32 = candidates.iter().map(|b| b.weight.max(1)).sum();
            let pick = rand::thread_rng().gen_range(0..total);
            let mut cumulative = 0u32;
            for b in candidates {
                cumulative += b.weight.max(1);
                if pick < cumulative {
                    return Some(Arc::clone(b));
                }
            }
            Some(Arc::clone(candidates.last().unwrap()))
        }
    }
}
