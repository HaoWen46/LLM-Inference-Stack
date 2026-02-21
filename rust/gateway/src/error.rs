use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

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

    #[error("Cannot reach upstream: {0}")]
    UpstreamError(String),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Service is shutting down")]
    ShuttingDown,

    #[error("Internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let status = match &self {
            GatewayError::Unauthorized => StatusCode::UNAUTHORIZED,
            GatewayError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::QuotaExceeded => StatusCode::TOO_MANY_REQUESTS,
            GatewayError::CircuitOpen => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::UpstreamTimeout => StatusCode::GATEWAY_TIMEOUT,
            GatewayError::UpstreamError(_) => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::BadRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::ShuttingDown => StatusCode::SERVICE_UNAVAILABLE,
            GatewayError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let message = match &self {
            GatewayError::Internal(_) => "Internal server error".to_string(),
            other => other.to_string(),
        };

        let mut headers = axum::http::HeaderMap::new();
        if matches!(self, GatewayError::Unauthorized) {
            headers.insert(
                axum::http::header::WWW_AUTHENTICATE,
                axum::http::HeaderValue::from_static("Bearer"),
            );
        }

        (status, headers, Json(json!({"detail": message}))).into_response()
    }
}
