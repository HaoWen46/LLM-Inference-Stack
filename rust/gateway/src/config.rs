use anyhow::{Context, Result};

use crate::backends::BackendDef;

/// All gateway configuration, loaded from environment variables at startup.
/// The shell (or Docker env_file) is responsible for setting these — no dotenv parsing here.
#[derive(Debug, Clone)]
pub struct Config {
    // ── Backends ──────────────────────────────────────────────────────────────
    /// Parsed from `BACKENDS` (multi-backend) or `VLLM_URL`/`VLLM_API_KEY` (legacy).
    pub backends: Vec<BackendDef>,

    // ── Default model name (used when request body omits the `model` field) ──
    pub served_model_name: String,

    // ── Gateway listen ────────────────────────────────────────────────────────
    pub gateway_host: String,
    pub gateway_port: u16,

    // ── Database ──────────────────────────────────────────────────────────────
    pub database_url: String,

    // ── Admin ──────────────────────────────────────────────────────────────────
    pub admin_key: String,

    // ── Rate limiting ─────────────────────────────────────────────────────────
    pub rate_limit_per_minute: u32,

    // ── Timeouts ──────────────────────────────────────────────────────────────
    pub request_timeout_secs: u64,
    pub connect_timeout_secs: u64,

    // ── Logging ───────────────────────────────────────────────────────────────
    pub log_level: String,

    // ── OpenTelemetry ─────────────────────────────────────────────────────────
    pub otel_exporter_otlp_endpoint: String,
    pub otel_enabled: bool,

    // ── Circuit breaker (shared defaults — applied per-backend) ───────────────
    pub cb_failure_threshold: u64,
    pub cb_recovery_timeout_secs: u64,
    pub cb_half_open_max_calls: u64,

    // ── Token quota (0 = unlimited) ───────────────────────────────────────────
    pub daily_token_quota: u64,

    // ── Local model cache ─────────────────────────────────────────────────────
    pub model_cache_dir: String,

    // ── LoRA aliases ──────────────────────────────────────────────────────────
    /// Alias names parsed from LORA_MODULES="alias1=path1 alias2=path2".
    /// Requests with model matching one of these are forwarded as-is (no rewrite).
    pub lora_aliases: Vec<String>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let database_url = std::env::var("DATABASE_URL")
            .context("DATABASE_URL is required")?;
        let admin_key = std::env::var("ADMIN_KEY")
            .context("ADMIN_KEY is required")?;

        // VLLM_API_KEY is required only when BACKENDS is not set.
        let vllm_api_key = std::env::var("VLLM_API_KEY").unwrap_or_default();
        let backends_str = env_or("BACKENDS", "");
        if backends_str.is_empty() && vllm_api_key.is_empty() {
            anyhow::bail!("VLLM_API_KEY is required when BACKENDS is not set");
        }

        let vllm_url = env_or("VLLM_URL", "http://localhost:8000");
        let served_model_name = env_or("SERVED_MODEL_NAME", "llama-7b");
        let backends = parse_backends(&backends_str, &vllm_url, &vllm_api_key, &served_model_name);

        Ok(Config {
            backends,
            served_model_name,
            gateway_host: env_or("GATEWAY_HOST", "0.0.0.0"),
            gateway_port: env_u16("GATEWAY_PORT", 8080),
            database_url,
            admin_key,
            rate_limit_per_minute: env_u32("RATE_LIMIT_PER_MINUTE", 60),
            request_timeout_secs: env_u64("REQUEST_TIMEOUT_SECONDS", 300),
            connect_timeout_secs: env_u64("CONNECT_TIMEOUT_SECONDS", 5),
            log_level: env_or("LOG_LEVEL", "INFO"),
            otel_exporter_otlp_endpoint: env_or(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://localhost:4317",
            ),
            otel_enabled: env_bool("OTEL_ENABLED", true),
            cb_failure_threshold: env_u64("CB_FAILURE_THRESHOLD", 5),
            cb_recovery_timeout_secs: env_u64("CB_RECOVERY_TIMEOUT", 30),
            cb_half_open_max_calls: env_u64("CB_HALF_OPEN_MAX_CALLS", 1),
            daily_token_quota: env_u64("DAILY_TOKEN_QUOTA", 0),
            model_cache_dir: env_or("MODEL_CACHE_DIR", "models"),
            lora_aliases: parse_lora_aliases(&env_or("LORA_MODULES", "")),
        })
    }
}

// ── Backend parsing ───────────────────────────────────────────────────────────

/// Parse `BACKENDS` into a list of `BackendDef`.
///
/// Format: semicolon-separated entries, each with pipe-separated fields:
///   `url|api_key|models|weight|served_name`
///
/// - `api_key`      — empty → use `fallback_api_key` (VLLM_API_KEY)
/// - `models`       — comma-separated; empty or `*` → catch-all backend
/// - `weight`       — positive integer for weighted random selection; default 1
/// - `served_name`  — the model name the backend expects in the request body;
///                    empty → no model-field rewrite
///
/// If `backends_str` is empty, creates a single backend from the legacy
/// `VLLM_URL` / `VLLM_API_KEY` / `SERVED_MODEL_NAME` env vars.
pub fn parse_backends(
    backends_str: &str,
    fallback_url: &str,
    fallback_api_key: &str,
    fallback_served_model: &str,
) -> Vec<BackendDef> {
    if backends_str.is_empty() {
        return vec![BackendDef {
            url: fallback_url.to_string(),
            api_key: fallback_api_key.to_string(),
            models: vec![fallback_served_model.to_string()],
            weight: 1,
            served_model_name: Some(fallback_served_model.to_string()),
        }];
    }

    backends_str
        .split(';')
        .filter(|s| !s.trim().is_empty())
        .map(|entry| {
            let parts: Vec<&str> = entry.splitn(5, '|').collect();
            let url = parts.get(0).unwrap_or(&"").trim().to_string();
            let api_key_raw = parts.get(1).unwrap_or(&"").trim();
            let api_key = if api_key_raw.is_empty() {
                fallback_api_key
            } else {
                api_key_raw
            }
            .to_string();
            let models_str = parts.get(2).unwrap_or(&"").trim();
            let models: Vec<String> = if models_str.is_empty() || models_str == "*" {
                vec![] // catch-all
            } else {
                models_str
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect()
            };
            let weight: u32 = parts
                .get(3)
                .and_then(|w| w.trim().parse().ok())
                .unwrap_or(1);
            let served_model_name = parts.get(4).and_then(|s| {
                let s = s.trim();
                if s.is_empty() { None } else { Some(s.to_string()) }
            });
            BackendDef { url, api_key, models, weight, served_model_name }
        })
        .collect()
}

// ── Env helpers ───────────────────────────────────────────────────────────────

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parse "alias1=path1 alias2=path2" → ["alias1", "alias2"]
fn parse_lora_aliases(lora_modules: &str) -> Vec<String> {
    lora_modules
        .split_whitespace()
        .filter_map(|entry| entry.split_once('=').map(|(alias, _)| alias.to_string()))
        .collect()
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(default)
}
