/// Batch inference API — create, poll, and cancel offline inference jobs.
///
/// Simplified from the OpenAI batch API: requests are sent inline instead of
/// via file_id, and results are retrieved directly from the batch object.
///
/// Endpoints (nested under /v1/batches):
///   POST   /          create_batch
///   GET    /          list_batches
///   GET    /:id       get_batch
///   POST   /:id/cancel cancel_batch
///   GET    /:id/results get_batch_results
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use std::sync::Arc;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{auth::ApiKey, error::GatewayError, AppState};

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateBatchRequest {
    pub requests: Vec<BatchItem>,
    #[allow(dead_code)]
    pub completion_window: Option<String>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct BatchItem {
    pub custom_id: String,
    pub method: String,
    pub url: String,
    pub body: Value,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", post(create_batch).get(list_batches))
        .route("/:id", get(get_batch))
        .route("/:id/cancel", post(cancel_batch))
        .route("/:id/results", get(get_batch_results))
}

// ── Handlers ──────────────────────────────────────────────────────────────────

pub async fn create_batch(
    State(state): State<Arc<AppState>>,
    ApiKey(api_key): ApiKey,
    Json(req): Json<CreateBatchRequest>,
) -> Result<impl IntoResponse, GatewayError> {
    if state.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::ShuttingDown);
    }
    if req.requests.is_empty() {
        return Err(GatewayError::BadRequest(
            "requests array must not be empty".into(),
        ));
    }
    if req.requests.len() > 1000 {
        return Err(GatewayError::BadRequest(
            "batch size must not exceed 1000 requests".into(),
        ));
    }
    for item in &req.requests {
        if item.method != "POST" {
            return Err(GatewayError::BadRequest(format!(
                "unsupported method '{}' in request '{}'",
                item.method, item.custom_id
            )));
        }
        if !matches!(
            item.url.as_str(),
            "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings"
        ) {
            return Err(GatewayError::BadRequest(format!(
                "unsupported url '{}' in request '{}'",
                item.url, item.custom_id
            )));
        }
    }

    let key_hash = hash_key(&api_key);
    let total = req.requests.len() as i32;
    let requests_text = serde_json::to_string(&req.requests)
        .map_err(|e| GatewayError::Internal(anyhow::anyhow!(e)))?;

    let pool = state.db_pool.clone();

    let batch_id: Uuid = sqlx::query_scalar(
        "INSERT INTO batches (key_hash, status, total_requests, requests) \
         VALUES ($1, 'validating', $2, $3) RETURNING id",
    )
    .bind(&key_hash)
    .bind(total)
    .bind(&requests_text)
    .fetch_one(&pool)
    .await
    .map_err(|e| GatewayError::Internal(anyhow::anyhow!(e)))?;

    tokio::spawn(batch_worker(batch_id, state.clone(), pool));

    info!(%batch_id, total, "batch created");

    let created_at = chrono::Utc::now().timestamp();
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": batch_id.to_string(),
            "object": "batch",
            "status": "validating",
            "total_requests": total,
            "completed_count": 0,
            "failed_count": 0,
            "created_at": created_at,
            "completed_at": null,
        })),
    ))
}

pub async fn get_batch(
    State(state): State<Arc<AppState>>,
    ApiKey(api_key): ApiKey,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, GatewayError> {
    let key_hash = hash_key(&api_key);
    let row = sqlx::query!(
        "SELECT id, status, total_requests, completed_count, failed_count, \
                created_at, completed_at \
         FROM batches WHERE id = $1 AND key_hash = $2",
        id,
        key_hash
    )
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| GatewayError::Internal(anyhow::anyhow!(e)))?
    .ok_or_else(|| GatewayError::NotFound(format!("batch {}", id)))?;

    Ok(Json(json!({
        "id": row.id.to_string(),
        "object": "batch",
        "status": row.status,
        "total_requests": row.total_requests,
        "completed_count": row.completed_count,
        "failed_count": row.failed_count,
        "created_at": row.created_at.timestamp(),
        "completed_at": row.completed_at.map(|t| t.timestamp()),
    })))
}

pub async fn list_batches(
    State(state): State<Arc<AppState>>,
    ApiKey(api_key): ApiKey,
) -> Result<impl IntoResponse, GatewayError> {
    let key_hash = hash_key(&api_key);
    let rows = sqlx::query!(
        "SELECT id, status, total_requests, completed_count, failed_count, \
                created_at, completed_at \
         FROM batches WHERE key_hash = $1 ORDER BY created_at DESC LIMIT 100",
        key_hash
    )
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| GatewayError::Internal(anyhow::anyhow!(e)))?;

    let data: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "id": r.id.to_string(),
                "object": "batch",
                "status": r.status,
                "total_requests": r.total_requests,
                "completed_count": r.completed_count,
                "failed_count": r.failed_count,
                "created_at": r.created_at.timestamp(),
                "completed_at": r.completed_at.map(|t| t.timestamp()),
            })
        })
        .collect();

    Ok(Json(json!({"object": "list", "data": data})))
}

pub async fn cancel_batch(
    State(state): State<Arc<AppState>>,
    ApiKey(api_key): ApiKey,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, GatewayError> {
    let key_hash = hash_key(&api_key);
    let updated = sqlx::query_scalar::<_, Uuid>(
        "UPDATE batches SET status = 'cancelled' \
         WHERE id = $1 AND key_hash = $2 AND status IN ('validating', 'in_progress') \
         RETURNING id",
    )
    .bind(id)
    .bind(&key_hash)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| GatewayError::Internal(anyhow::anyhow!(e)))?;

    if updated.is_none() {
        return Err(GatewayError::NotFound(format!(
            "batch {} not found or not cancellable",
            id
        )));
    }

    info!(%id, "batch cancelled");
    Ok(Json(json!({
        "id": id.to_string(),
        "object": "batch",
        "status": "cancelled",
    })))
}

pub async fn get_batch_results(
    State(state): State<Arc<AppState>>,
    ApiKey(api_key): ApiKey,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, GatewayError> {
    let key_hash = hash_key(&api_key);
    let row = sqlx::query!(
        "SELECT status, results FROM batches WHERE id = $1 AND key_hash = $2",
        id,
        key_hash
    )
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| GatewayError::Internal(anyhow::anyhow!(e)))?
    .ok_or_else(|| GatewayError::NotFound(format!("batch {}", id)))?;

    if row.status != "completed" && row.status != "failed" {
        return Err(GatewayError::BadRequest(format!(
            "batch {} is not complete (status: {})",
            id, row.status
        )));
    }

    let results: Value = serde_json::from_str(&row.results)
        .unwrap_or_else(|_| json!([]));

    Ok(Json(json!({
        "id": id.to_string(),
        "object": "batch.results",
        "data": results,
    })))
}

// ── Background worker ─────────────────────────────────────────────────────────

async fn batch_worker(batch_id: Uuid, state: Arc<AppState>, pool: PgPool) {
    // Transition: validating → in_progress (skip if already cancelled)
    if let Err(e) = sqlx::query(
        "UPDATE batches SET status = 'in_progress' WHERE id = $1 AND status = 'validating'",
    )
    .bind(batch_id)
    .execute(&pool)
    .await
    {
        error!(%batch_id, error = %e, "batch: failed to mark in_progress");
        return;
    }

    // Load requests text
    let requests_text: String = match sqlx::query_scalar(
        "SELECT requests FROM batches WHERE id = $1",
    )
    .bind(batch_id)
    .fetch_one(&pool)
    .await
    {
        Ok(t) => t,
        Err(e) => {
            error!(%batch_id, error = %e, "batch: failed to load requests");
            return;
        }
    };

    let items: Vec<BatchItem> = match serde_json::from_str(&requests_text) {
        Ok(v) => v,
        Err(e) => {
            error!(%batch_id, error = %e, "batch: failed to parse requests");
            let _ = sqlx::query("UPDATE batches SET status = 'failed' WHERE id = $1")
                .bind(batch_id)
                .execute(&pool)
                .await;
            return;
        }
    };

    let mut results: Vec<Value> = Vec::with_capacity(items.len());
    let mut completed: i32 = 0;
    let mut failed: i32 = 0;

    for item in &items {
        // Check for cancellation
        let cur_status: Option<String> =
            sqlx::query_scalar("SELECT status FROM batches WHERE id = $1")
                .bind(batch_id)
                .fetch_optional(&pool)
                .await
                .unwrap_or(None);

        if cur_status.as_deref() == Some("cancelled") {
            info!(%batch_id, "batch cancelled — worker stopping");
            return;
        }

        if state
            .shutting_down
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            warn!(%batch_id, "gateway shutting down — batch worker stopping early");
            break;
        }

        // Build vLLM request body
        let mut body = item.body.clone();
        body["stream"] = Value::Bool(false);

        // Rewrite model for generation endpoints (not embeddings, honour LoRA aliases)
        if item.url != "/v1/embeddings" {
            let req_model = body
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !state.config.lora_aliases.contains(&req_model) {
                body["model"] = Value::String(state.config.served_model_name.clone());
            }
        }

        let vllm_url = format!("{}{}", state.config.vllm_url, item.url);

        let result = match state
            .client
            .post(&vllm_url)
            .header(
                "Authorization",
                format!("Bearer {}", state.config.vllm_api_key),
            )
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                let status_code = resp.status().as_u16();
                let is_ok = resp.status().is_success();
                match resp.json::<Value>().await {
                    Ok(resp_body) => {
                        if is_ok {
                            completed += 1;
                            json!({
                                "custom_id": item.custom_id,
                                "status_code": status_code,
                                "body": resp_body,
                                "error": null,
                            })
                        } else {
                            failed += 1;
                            json!({
                                "custom_id": item.custom_id,
                                "status_code": status_code,
                                "body": null,
                                "error": resp_body,
                            })
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        json!({
                            "custom_id": item.custom_id,
                            "status_code": status_code,
                            "body": null,
                            "error": { "message": e.to_string() },
                        })
                    }
                }
            }
            Err(e) => {
                failed += 1;
                json!({
                    "custom_id": item.custom_id,
                    "status_code": null,
                    "body": null,
                    "error": { "message": e.to_string() },
                })
            }
        };

        results.push(result);

        // Persist progress after each item
        let _ = sqlx::query(
            "UPDATE batches SET completed_count = $1, failed_count = $2 WHERE id = $3",
        )
        .bind(completed)
        .bind(failed)
        .bind(batch_id)
        .execute(&pool)
        .await;
    }

    let final_status = if results.is_empty() || (failed == results.len() as i32) {
        "failed"
    } else {
        "completed"
    };

    let results_text =
        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());

    let _ = sqlx::query(
        "UPDATE batches \
         SET status = $1, completed_count = $2, failed_count = $3, \
             results = $4, completed_at = NOW() \
         WHERE id = $5",
    )
    .bind(final_status)
    .bind(completed)
    .bind(failed)
    .bind(&results_text)
    .bind(batch_id)
    .execute(&pool)
    .await;

    info!(%batch_id, completed, failed, final_status, "batch worker complete");
}

// ── Utility ───────────────────────────────────────────────────────────────────

fn hash_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    format!("{:x}", hasher.finalize())
}
