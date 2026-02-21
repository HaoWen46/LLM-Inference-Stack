use anyhow::{Context, Result};
use std::collections::HashSet;

/// All gateway configuration, loaded from environment variables at startup.
/// The shell (or Docker env_file) is responsible for setting these — no dotenv parsing here.
#[derive(Debug, Clone)]
pub struct Config {
    // ── Upstream vLLM ───────────────────────────────────────────────────────
    pub vllm_url: String,
    pub vllm_api_key: String,
    pub served_model_name: String,

    // ── Gateway listen ──────────────────────────────────────────────────────
    pub gateway_host: String,
    pub gateway_port: u16,

    // ── Auth ─────────────────────────────────────────────────────────────────
    pub gateway_api_keys: HashSet<String>,

    // ── Rate limiting ────────────────────────────────────────────────────────
    pub rate_limit_per_minute: u32,

    // ── Timeouts ─────────────────────────────────────────────────────────────
    pub request_timeout_secs: u64,
    pub connect_timeout_secs: u64,

    // ── Logging ──────────────────────────────────────────────────────────────
    pub log_level: String,

    // ── OpenTelemetry ────────────────────────────────────────────────────────
    pub otel_exporter_otlp_endpoint: String,
    pub otel_enabled: bool,

    // ── Circuit breaker ──────────────────────────────────────────────────────
    pub cb_failure_threshold: u64,
    pub cb_recovery_timeout_secs: u64,
    pub cb_half_open_max_calls: u64,

    // ── Token quota (0 = unlimited) ──────────────────────────────────────────
    pub daily_token_quota: u64,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let vllm_api_key = std::env::var("VLLM_API_KEY")
            .context("VLLM_API_KEY is required")?;

        let raw_keys = env_or("GATEWAY_API_KEYS", "dev-key-1");
        let gateway_api_keys: HashSet<String> = raw_keys
            .split(',')
            .filter_map(|k| {
                let k = k.trim().to_string();
                if k.is_empty() { None } else { Some(k) }
            })
            .collect();

        Ok(Config {
            vllm_url: env_or("VLLM_URL", "http://localhost:8000"),
            vllm_api_key,
            served_model_name: env_or("SERVED_MODEL_NAME", "llama-7b"),
            gateway_host: env_or("GATEWAY_HOST", "0.0.0.0"),
            gateway_port: env_u16("GATEWAY_PORT", 8080),
            gateway_api_keys,
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
        })
    }
}

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

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(default)
}
