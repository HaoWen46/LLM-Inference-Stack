use axum::{
    extract::FromRequestParts,
    http::request::Parts,
};
use std::sync::Arc;

use crate::{error::GatewayError, keys::sha256_hex, AppState};

/// Axum extractor that validates the Bearer token or `x-api-key` header.
/// Injects the raw (plaintext) API key for downstream quota tracking.
/// Hot path: SHA-256 hash then DashMap lookup — no database I/O, no await on DB.
pub struct ApiKey(pub String);

#[axum::async_trait]
impl FromRequestParts<Arc<AppState>> for ApiKey {
    type Rejection = GatewayError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        let raw_key = match extract_token(parts) {
            Some(k) => k,
            None => {
                state.metrics.auth_failures.inc();
                return Err(GatewayError::Unauthorized);
            }
        };

        let hash = sha256_hex(&raw_key);
        match state.key_store.lookup(&hash) {
            Some(cached) => {
                // Reject expired keys
                if let Some(exp) = cached.expires_at {
                    if exp < chrono::Utc::now() {
                        state.metrics.auth_failures.inc();
                        return Err(GatewayError::Unauthorized);
                    }
                }
                // Fire-and-forget last_used_at update (does not block the hot path)
                state.key_store.touch_last_used(cached.id);
                Ok(ApiKey(raw_key))
            }
            None => {
                state.metrics.auth_failures.inc();
                Err(GatewayError::Unauthorized)
            }
        }
    }
}

/// Extract the API key from `Authorization: Bearer <token>` or `x-api-key` header.
pub fn extract_token(parts: &Parts) -> Option<String> {
    // Try Authorization: Bearer <token>
    if let Some(auth) = parts.headers.get("authorization") {
        if let Ok(s) = auth.to_str() {
            if let Some(token) = s.strip_prefix("Bearer ") {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Some(token);
                }
            }
        }
    }
    // Try x-api-key header (used by some OpenAI-compatible clients)
    parts
        .headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
