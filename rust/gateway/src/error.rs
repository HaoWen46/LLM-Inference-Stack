use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;
use tracing::error;

#[derive(Error, Debug)]
pub enum GatewayError {
    #[error("Invalid or missing API key")]
    Unauthorized,

    #[error("Rate limit exceeded")]
    RateLimited,

    #[error("Daily token quota exceeded")]
    QuotaExceeded,

    #[error("Circuit breaker is open — upstream considered unhealthy")]
    CircuitOpen,

    #[error("Upstream request timed out")]
    UpstreamTimeout,

    // Internal detail kept in the variant but never sent to the client.
    #[error("upstream error: {0}")]
    UpstreamError(String),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Service is shutting down")]
    ShuttingDown,

    #[error("Internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        // Log full detail internally before sanitizing the client-facing message.
        match &self {
            GatewayError::UpstreamError(detail) => {
                error!(detail, "upstream error");
            }
            GatewayError::Internal(e) => {
                error!(detail = %e, "internal error");
            }
            _ => {}
        }

        let status = match &self {
            GatewayError::Unauthorized => StatusCode::UNAUTHORIZED,
            GatewayError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::QuotaExceeded => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::CircuitOpen => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::UpstreamTimeout => StatusCode::GATEWAY_TIMEOUT,
            GatewayError::UpstreamError(_) => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::BadRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::NotFound(_) => StatusCode::NOT_FOUND,
            GatewayError::Forbidden(_) => StatusCode::FORBIDDEN,
            GatewayError::ShuttingDown => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        // Sanitized messages — no internal details (IPs, socket errors, etc.) leak.
        let (message, error_type) = match &self {
            GatewayError::Unauthorized => (
                "Invalid or missing API key".to_string(),
                "authentication_error",
            ),
            GatewayError::RateLimited => (
                "Rate limit exceeded".to_string(),
                "rate_limit_error",
            ),
            GatewayError::QuotaExceeded => (
                "Daily token quota exceeded".to_string(),
                "rate_limit_error",
            ),
            GatewayError::BadRequest(msg) => (
                format!("Invalid request: {}", msg),
                "invalid_request_error",
            ),
            GatewayError::NotFound(msg) => (
                format!("Not found: {}", msg),
                "invalid_request_error",
            ),
            GatewayError::Forbidden(msg) => (
                format!("Forbidden: {}", msg),
                "invalid_request_error",
            ),
            GatewayError::UpstreamError(_) => (
                "Upstream service unavailable".to_string(),
                "api_error",
            ),
            GatewayError::Internal(_) => (
                "Internal server error".to_string(),
                "api_error",
            ),
            other => (other.to_string(), "api_error"),
        };

        let mut headers = axum::http::HeaderMap::new();
        if matches!(self, GatewayError::Unauthorized) {
            headers.insert(
                axum::http::header::WWW_AUTHENTICATE,
                axum::http::HeaderValue::from_static("Bearer"),
            );
        }

        (
            status,
            headers,
            Json(json!({
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": null,
                    "code": null
                }
            })),
        )
            .into_response()
    }
}
