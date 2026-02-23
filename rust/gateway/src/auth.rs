use axum::{
    extract::FromRequestParts,
    http::request::Parts,
};
use std::{collections::HashSet, sync::Arc};
use subtle::ConstantTimeEq;

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
            Some(key) => match ct_key_lookup(&key, &state.config.gateway_api_keys) {
                Some(matched) => Ok(ApiKey(matched)),
                None => {
                    state.metrics.auth_failures.inc();
                    Err(GatewayError::Unauthorized)
                }
            },
            None => {
                state.metrics.auth_failures.inc();
                Err(GatewayError::Unauthorized)
            }
        }
    }
}

/// Constant-time key lookup: iterates over ALL keys without short-circuiting
/// to prevent timing side-channels. Returns the matched stored key or None.
fn ct_key_lookup(submitted: &str, valid_keys: &HashSet<String>) -> Option<String> {
    let submitted_bytes = submitted.as_bytes();
    let mut result: Option<String> = None;
    let mut found = 0u8;

    for stored in valid_keys {
        let stored_bytes = stored.as_bytes();
        // ct_eq requires same-length slices; length mismatch → 0 (constant-time).
        let eq: u8 = if submitted_bytes.len() == stored_bytes.len() {
            submitted_bytes.ct_eq(stored_bytes).unwrap_u8()
        } else {
            0u8
        };
        // Accumulate without branching on eq so all keys are always compared.
        found |= eq;
        // Record the match; the branch only runs for the correct key.
        if eq == 1 && result.is_none() {
            result = Some(stored.clone());
        }
    }

    if found == 1 { result } else { None }
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
