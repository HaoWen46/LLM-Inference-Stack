/// Admin REST API for API key management and org/usage queries.
///
/// All endpoints require `Authorization: Bearer <ADMIN_KEY>` where ADMIN_KEY
/// is a separate secret from the gateway API keys.
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, patch, post, put},
    Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    auth::extract_token,
    error::GatewayError,
    keys::{CreateKeyOptions, KeyLimits},
    AppState,
};

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

// ── Request / response types — keys ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub label: String,
    pub expires_at: Option<DateTime<Utc>>,
    pub rpm_limit: Option<i32>,
    pub daily_token_limit: Option<i64>,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
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
    pub rpm_limit: Option<i32>,
    pub daily_token_limit: Option<i64>,
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
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

/// PUT /admin/keys/:id/limits — replace all per-key limit fields (PUT semantics).
/// Set a field to `null` to clear it (revert to global default).
#[derive(Deserialize)]
pub struct UpdateLimitsRequest {
    pub expires_at: Option<DateTime<Utc>>,
    pub rpm_limit: Option<i32>,
    pub daily_token_limit: Option<i64>,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
}

// ── Request / response types — orgs ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateOrgRequest {
    pub name: String,
}

#[derive(Serialize)]
pub struct OrgResponse {
    pub id: Uuid,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize)]
pub struct OrgUsageResponse {
    pub org_id: Uuid,
    pub date: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub request_count: i64,
    pub key_count: i64,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        // Key CRUD
        .route("/keys", post(create_key).get(list_keys))
        .route("/keys/:id", patch(patch_key).delete(delete_key))
        .route("/keys/:id/limits", put(update_limits))
        // Org CRUD + usage
        .route("/orgs", post(create_org).get(list_orgs))
        .route("/orgs/:id/usage", get(org_usage))
}

// ── Key handlers ──────────────────────────────────────────────────────────────

async fn create_key(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateKeyRequest>,
) -> Result<impl IntoResponse, GatewayError> {
    let opts = CreateKeyOptions {
        label: req.label.clone(),
        expires_at: req.expires_at,
        rpm_limit: req.rpm_limit,
        daily_token_limit: req.daily_token_limit,
        allowed_models: req.allowed_models,
        org_id: req.org_id,
    };
    let (id, created_at, plaintext) = state.key_store.create(opts).await?;
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
            rpm_limit: k.rpm_limit,
            daily_token_limit: k.daily_token_limit,
            allowed_models: k.allowed_models,
            org_id: k.org_id,
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
        return Err(GatewayError::NotFound(format!("key {}", id)));
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
        return Err(GatewayError::NotFound(format!("key {}", id)));
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn update_limits(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateLimitsRequest>,
) -> Result<impl IntoResponse, GatewayError> {
    let limits = KeyLimits {
        expires_at: req.expires_at,
        rpm_limit: req.rpm_limit,
        daily_token_limit: req.daily_token_limit,
        allowed_models: req.allowed_models,
        org_id: req.org_id,
    };
    let found = state.key_store.update_limits(id, limits).await?;
    if !found {
        return Err(GatewayError::NotFound(format!("key {}", id)));
    }
    Ok(StatusCode::NO_CONTENT)
}

// ── Org handlers ──────────────────────────────────────────────────────────────

async fn create_org(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateOrgRequest>,
) -> Result<impl IntoResponse, GatewayError> {
    let org = state.key_store.create_org(&req.name).await?;
    Ok((
        StatusCode::CREATED,
        Json(OrgResponse { id: org.id, name: org.name, created_at: org.created_at }),
    ))
}

async fn list_orgs(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, GatewayError> {
    let orgs = state.key_store.list_orgs().await?;
    let data: Vec<OrgResponse> = orgs
        .into_iter()
        .map(|o| OrgResponse { id: o.id, name: o.name, created_at: o.created_at })
        .collect();
    Ok(Json(data))
}

async fn org_usage(
    _auth: AdminAuth,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, GatewayError> {
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let usage = state.key_store.get_org_usage(id, &today).await?;
    Ok(Json(OrgUsageResponse {
        org_id: id,
        date: usage.date,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        request_count: usage.request_count,
        key_count: usage.key_count,
    }))
}
