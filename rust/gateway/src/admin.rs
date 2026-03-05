/// Admin REST API for API key management.
///
/// All endpoints require `Authorization: Bearer <ADMIN_KEY>` where ADMIN_KEY
/// is a separate secret from the gateway API keys.
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{patch, post},
    Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::{auth::extract_token, error::GatewayError, AppState};

// ── Admin auth extractor ──────────────────────────────────────────────────────

pub struct AdminAuth;

#[axum::async_trait]
impl axum::extract::FromRequestParts<Arc<AppState>> for AdminAuth {
    type Rejection = GatewayError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        match extract_token(parts) {
            Some(key) if key == state.config.admin_key => Ok(AdminAuth),
            _ => Err(GatewayError::Unauthorized),
        }
    }
}

// ── Request / response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub label: String,
}

#[derive(Serialize)]
pub struct CreateKeyResponse {
    pub id: Uuid,
    pub label: String,
    /// Plaintext key — shown exactly once.
    pub key: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize)]
pub struct KeyListEntry {
    pub id: Uuid,
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub enabled: bool,
    pub last_used_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize)]
pub struct PatchKeyRequest {
    pub enabled: bool,
}

#[derive(Serialize)]
pub struct PatchKeyResponse {
    pub id: Uuid,
    pub enabled: bool,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/keys", post(create_key).get(list_keys))
        .route("/keys/:id", patch(patch_key).delete(delete_key))
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn create_key(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateKeyRequest>,
) -> Result<impl IntoResponse, GatewayError> {
    let (id, created_at, plaintext) = state.key_store.create(&req.label).await?;
    Ok((
        StatusCode::CREATED,
        Json(CreateKeyResponse {
            id,
            label: req.label,
            key: plaintext,
            created_at,
        }),
    ))
}

async fn list_keys(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, GatewayError> {
    let keys = state.key_store.list().await?;
    let entries: Vec<KeyListEntry> = keys
        .into_iter()
        .map(|k| KeyListEntry {
            id: k.id,
            label: k.label,
            created_at: k.created_at,
            expires_at: k.expires_at,
            enabled: k.enabled,
            last_used_at: k.last_used_at,
        })
        .collect();
    Ok(Json(entries))
}

async fn patch_key(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<PatchKeyRequest>,
) -> Result<impl IntoResponse, GatewayError> {
    let found = state.key_store.set_enabled(id, req.enabled).await?;
    if !found {
        return Err(GatewayError::BadRequest(format!("key {} not found", id)));
    }
    Ok(Json(PatchKeyResponse { id, enabled: req.enabled }))
}

async fn delete_key(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, GatewayError> {
    let found = state.key_store.delete(id).await?;
    if !found {
        return Err(GatewayError::BadRequest(format!("key {} not found", id)));
    }
    Ok(StatusCode::NO_CONTENT)
}
