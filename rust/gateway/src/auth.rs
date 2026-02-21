use axum::{
    extract::FromRequestParts,
    http::request::Parts,
};
use std::sync::Arc;

use crate::{error::GatewayError, AppState};

/// Axum extractor that validates the Bearer token or `x-api-key` header.
/// Injects the raw (plaintext) API key for downstream quota tracking.
pub struct ApiKey(pub String);

#[axum::async_trait]
impl FromRequestParts<Arc<AppState>> for ApiKey {
    type Rejection = GatewayError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        match extract_token(parts) {
            Some(key) if state.config.gateway_api_keys.contains(&key) => Ok(ApiKey(key)),
            _ => {
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
