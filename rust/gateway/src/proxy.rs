/// Upstream proxy handlers: sync (buffered) and streaming (SSE passthrough).
use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::{collections::HashSet, net::SocketAddr, path::Path, sync::Arc, time::Instant};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

use crate::{
    auth::ApiKey,
    backends::Backend,
    error::GatewayError,
    keys::sha256_hex,
    metrics::{EndpointModelLabels, ModelLabel, RequestLabels, StatusCodeLabel},
    AppState,
};

// ── Public route handlers ────────────────────────────────────────────────────

/// GET /v1/models/local — list model weights available on disk (no vLLM call)
pub async fn local_models_handler(
    State(state): State<Arc<AppState>>,
    ApiKey(_api_key): ApiKey,
) -> Result<impl IntoResponse, GatewayError> {
    let root = Path::new(&state.config.model_cache_dir);
    let mut data: Vec<Value> = Vec::new();

    if root.exists() {
        // GGUF files — walk recursively, return path relative to cache_dir
        if let Ok(entries) = walkdir_gguf(root) {
            for rel_path in entries {
                data.push(serde_json::json!({
                    "id": rel_path,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "format": "gguf"
                }));
            }
        }

        // HuggingFace model directories — two layouts:
        //   1. HF Hub cache:  "models--org--repo"  → "org/repo"
        //   2. Flat snapshot: "org/repo"           → "org/repo"
        if let Ok(dir_iter) = std::fs::read_dir(root) {
            let mut hf_entries: Vec<String> = Vec::new();
            for entry in dir_iter.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = entry.file_name().into_string().unwrap_or_default();
                // Layout 1: models--org--repo
                if let Some(tail) = name.strip_prefix("models--") {
                    if let Some((org, repo)) = tail.split_once("--") {
                        hf_entries.push(format!("{}/{}", org, repo));
                        continue;
                    }
                }
                // Layout 2: org/repo — org dir containing subdirs with config.json
                if !name.starts_with('.') {
                    if let Ok(subdirs) = std::fs::read_dir(&path) {
                        for sub in subdirs.flatten() {
                            if sub.path().join("config.json").exists() {
                                let repo = sub.file_name().into_string().unwrap_or_default();
                                hf_entries.push(format!("{}/{}", name, repo));
                            }
                        }
                    }
                }
            }
            hf_entries.sort();
            hf_entries.dedup();
            for hf_id in hf_entries {
                data.push(serde_json::json!({
                    "id": hf_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "format": "hf"
                }));
            }
        }
    }

    Ok(Json(serde_json::json!({"object": "list", "data": data})))
}

/// Recursively collect `.gguf` paths relative to `root`, sorted.
fn walkdir_gguf(root: &Path) -> std::io::Result<Vec<String>> {
    let mut results = Vec::new();
    collect_gguf(root, root, &mut results)?;
    results.sort();
    Ok(results)
}

fn collect_gguf(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(root, &path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().into_owned());
            }
        }
    }
    Ok(())
}

/// GET /v1/usage — per-key daily token usage
pub async fn usage_handler(
    State(state): State<Arc<AppState>>,
    ApiKey(api_key): ApiKey,
) -> Result<impl IntoResponse, GatewayError> {
    if state.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::ShuttingDown);
    }

    let info = state
        .quota
        .get_usage(&api_key)
        .await
        .map_err(GatewayError::Internal)?;

    // Effective quota: per-key override if set, else global
    let key_hash = sha256_hex(&api_key);
    let effective_quota = state
        .key_store
        .lookup(&key_hash)
        .and_then(|m| m.daily_token_limit.map(|l| l as u64))
        .unwrap_or(state.config.daily_token_quota);
    let used = (info.prompt_tokens + info.completion_tokens) as u64;
    let remaining: i64 = if effective_quota == 0 {
        -1
    } else {
        (effective_quota as i64 - used as i64).max(0)
    };

    Ok(Json(serde_json::json!({
        "date":               info.date,
        "prompt_tokens":      info.prompt_tokens,
        "completion_tokens":  info.completion_tokens,
        "requests":           info.request_count,
        "quota":              effective_quota,
        "quota_remaining":    remaining,
    })))
}

/// GET /v1/models — fan-out to all healthy backends, merge and deduplicate results.
pub async fn models_handler(
    State(state): State<Arc<AppState>>,
    ApiKey(_api_key): ApiKey,
) -> Result<impl IntoResponse, GatewayError> {
    if state.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::ShuttingDown);
    }

    let backends = state.backend_router.healthy_backends();
    if backends.is_empty() {
        return Err(GatewayError::UpstreamError("no healthy backends available".into()));
    }

    let mut all_data: Vec<Value> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    for backend in &backends {
        let result = state
            .client
            .get(format!("{}/v1/models", backend.url))
            .header("Authorization", format!("Bearer {}", backend.api_key))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(bytes) = resp.bytes().await {
                    if let Ok(json) = serde_json::from_slice::<Value>(&bytes) {
                        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                            for item in data {
                                let id = item
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if seen_ids.insert(id) {
                                    all_data.push(item.clone());
                                }
                            }
                        }
                    }
                }
            }
            Ok(resp) => {
                warn!(url = %backend.url, status = resp.status().as_u16(), "models fanout: backend error");
            }
            Err(e) => {
                warn!(url = %backend.url, error = %e, "models fanout: backend unreachable");
            }
        }
    }

    Ok(Json(serde_json::json!({"object": "list", "data": all_data})))
}

/// POST /v1/chat/completions
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ApiKey(api_key): ApiKey,
    body: Bytes,
) -> Result<Response, GatewayError> {
    proxy_request(state, api_key, addr.ip(), body, "/v1/chat/completions").await
}

/// POST /v1/completions
pub async fn completions_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ApiKey(api_key): ApiKey,
    body: Bytes,
) -> Result<Response, GatewayError> {
    proxy_request(state, api_key, addr.ip(), body, "/v1/completions").await
}

// ── Core proxy dispatcher ────────────────────────────────────────────────────

async fn proxy_request(
    state: Arc<AppState>,
    api_key: String,
    client_ip: std::net::IpAddr,
    raw_body: Bytes,
    path: &'static str,
) -> Result<Response, GatewayError> {
    if state
        .shutting_down
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        return Err(GatewayError::ShuttingDown);
    }

    // ── Per-key metadata (one DashMap lookup — O(1), no await) ───────────────
    let key_hash = sha256_hex(&api_key);
    let key_meta = state.key_store.lookup(&key_hash);

    // ── Rate limit (per-IP GCRA) ─────────────────────────────────────────────
    if !state.rate_limiters.check(client_ip) {
        state.metrics.rate_limited.inc();
        return Err(GatewayError::RateLimited);
    }

    // ── Per-key RPM limit ────────────────────────────────────────────────────
    if let Some(ref meta) = key_meta {
        if let Some(rpm) = meta.rpm_limit {
            if !state.rate_limiters.check_key(&key_hash, rpm as u32) {
                state.metrics.rate_limited.inc();
                return Err(GatewayError::RateLimited);
            }
        }
    }

    // ── Parse request body ───────────────────────────────────────────────────
    let mut body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| GatewayError::BadRequest(e.to_string()))?;

    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.config.served_model_name)
        .to_string();

    // ── Model allowlist ──────────────────────────────────────────────────────
    if let Some(ref meta) = key_meta {
        if !meta.allowed_models.is_empty() && !meta.allowed_models.contains(&model) {
            return Err(GatewayError::Forbidden(format!(
                "model '{}' is not in the allowlist for this API key",
                model
            )));
        }
    }

    // ── Daily quota check (per-key override or global) ───────────────────────
    let per_key_limit = key_meta.as_ref().and_then(|m| m.daily_token_limit.map(|l| l as u64));
    if !state.quota.is_allowed(&api_key, per_key_limit) {
        state.metrics.quota_exceeded.inc();
        return Err(GatewayError::QuotaExceeded);
    }

    // ── Backend selection ────────────────────────────────────────────────────
    let backend = state
        .backend_router
        .select(&model)
        .ok_or_else(|| GatewayError::UpstreamError(
            format!("no available backend for model '{}'", model),
        ))?;

    // ── Model rewrite: honour LoRA aliases; otherwise use backend's served name ──
    if !state.config.lora_aliases.contains(&model) {
        if let Some(ref served_name) = backend.served_model_name {
            body["model"] = Value::String(served_name.clone());
        }
    }

    state.metrics.active_requests.inc();
    state.metrics.inflight_requests.inc();

    let result = if stream {
        stream_proxy(state.clone(), api_key, body, &model, path, backend).await
    } else {
        sync_proxy(state.clone(), api_key, body, &model, path, backend).await
    };

    state.metrics.active_requests.dec();
    state.metrics.inflight_requests.dec();

    result
}

// ── Sync (buffered) proxy ─────────────────────────────────────────────────────

async fn sync_proxy(
    state: Arc<AppState>,
    api_key: String,
    body: Value,
    model: &str,
    path: &str,
    backend: Arc<Backend>,
) -> Result<Response, GatewayError> {
    let start = Instant::now();

    backend.circuit_breaker.before_call()?;

    let upstream_url = format!("{}{}", backend.url, path);

    let resp = state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", backend.api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            backend.circuit_breaker.on_failure();
            if e.is_timeout() {
                GatewayError::UpstreamTimeout
            } else {
                GatewayError::UpstreamError(e.to_string())
            }
        })?;

    let status = resp.status();
    let duration = start.elapsed().as_secs_f64();

    if status.is_server_error() {
        backend.circuit_breaker.on_failure();
        state
            .metrics
            .upstream_errors
            .get_or_create(&StatusCodeLabel {
                status_code: status.as_str().to_string(),
            })
            .inc_by(1);
        warn!(status = status.as_u16(), url = %backend.url, "upstream error");
    } else {
        backend.circuit_breaker.on_success();
    }

    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    let (prompt_tokens, completion_tokens) =
        if let Ok(parsed) = serde_json::from_slice::<Value>(&body_bytes) {
            let usage = parsed.get("usage");
            let prompt = usage
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let completion = usage
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            (prompt, completion)
        } else {
            (0u64, 0u64)
        };

    state
        .metrics
        .request_count
        .get_or_create(&RequestLabels {
            method: "POST".into(),
            endpoint: path.to_string(),
            status_code: status.as_str().to_string(),
            model: model.to_string(),
        })
        .inc_by(1);

    state
        .metrics
        .request_latency
        .get_or_create(&EndpointModelLabels {
            endpoint: path.to_string(),
            model: model.to_string(),
        })
        .observe(duration);

    if prompt_tokens > 0 {
        state
            .metrics
            .tokens_prompted
            .get_or_create(&ModelLabel { model: model.to_string() })
            .inc_by(prompt_tokens);
    }
    if completion_tokens > 0 {
        state
            .metrics
            .tokens_generated
            .get_or_create(&ModelLabel { model: model.to_string() })
            .inc_by(completion_tokens);
    }

    info!(
        status = status.as_u16(),
        duration_ms = (duration * 1000.0) as u64,
        prompt_tokens,
        completion_tokens,
        backend = %backend.url,
        "request complete"
    );

    if prompt_tokens > 0 || completion_tokens > 0 {
        state.quota.add_tokens(&api_key, prompt_tokens, completion_tokens);
    }

    let axum_status =
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    Ok(Response::builder()
        .status(axum_status)
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap())
}

// ── Streaming SSE proxy ───────────────────────────────────────────────────────

async fn stream_proxy(
    state: Arc<AppState>,
    api_key: String,
    mut body: Value,
    model: &str,
    path: &str,
    backend: Arc<Backend>,
) -> Result<Response, GatewayError> {
    let start = Instant::now();
    body["stream"] = Value::Bool(true);
    // Ask vLLM to append a final usage chunk so we get exact token counts.
    body["stream_options"] = serde_json::json!({"include_usage": true});

    backend.circuit_breaker.before_call()?;

    let upstream_url = format!("{}{}", backend.url, path);
    let model = model.to_string();
    let path_str = path.to_string();

    // Channel: background task → axum response body
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    tokio::spawn({
        let state = state.clone();
        async move {
            let resp_result = state
                .client
                .post(&upstream_url)
                .header("Authorization", format!("Bearer {}", backend.api_key))
                .json(&body)
                .send()
                .await;

            let resp = match resp_result {
                Ok(r) => r,
                Err(e) => {
                    backend.circuit_breaker.on_failure();
                    let msg = if e.is_timeout() {
                        "upstream timeout"
                    } else {
                        "upstream connection error"
                    };
                    let _ = tx
                        .send(Ok(Bytes::from(format!(
                            "data: {{\"error\": \"{}\"}}\n\n",
                            msg
                        ))))
                        .await;
                    return;
                }
            };

            let status = resp.status();

            if status.is_server_error() {
                backend.circuit_breaker.on_failure();
                state
                    .metrics
                    .upstream_errors
                    .get_or_create(&StatusCodeLabel {
                        status_code: status.as_str().to_string(),
                    })
                    .inc_by(1);
                if let Ok(body_bytes) = resp.bytes().await {
                    let _ = tx.send(Ok(body_bytes)).await;
                }
                return;
            }

            backend.circuit_breaker.on_success();

            let mut byte_stream = resp.bytes_stream();
            let mut line_buf: Vec<u8> = Vec::new();
            let mut first_token = true;
            let mut ttft = 0f64;
            let mut exact_prompt_tokens: u64 = 0;
            let mut exact_completion_tokens: u64 = 0;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        error!(error = %e, "stream chunk error");
                        break;
                    }
                };

                line_buf.extend_from_slice(&chunk);

                loop {
                    match line_buf.iter().position(|&b| b == b'\n') {
                        None => break,
                        Some(nl_pos) => {
                            let line_bytes: Vec<u8> = line_buf.drain(..=nl_pos).collect();
                            let line = std::str::from_utf8(&line_bytes)
                                .unwrap_or("")
                                .trim_end_matches(['\n', '\r']);

                            if line.is_empty() {
                                continue;
                            }

                            if line.starts_with("data: ") && line != "data: [DONE]" {
                                if let Ok(chunk_json) =
                                    serde_json::from_str::<Value>(&line[6..])
                                {
                                    if first_token
                                        && chunk_json
                                            .pointer("/choices/0/delta/content")
                                            .is_some()
                                    {
                                        ttft = start.elapsed().as_secs_f64();
                                        state
                                            .metrics
                                            .ttft
                                            .get_or_create(&ModelLabel { model: model.clone() })
                                            .observe(ttft);
                                        info!(ttft_ms = (ttft * 1000.0) as u64, "first token");
                                        first_token = false;
                                    }

                                    if let Some(usage) = chunk_json.get("usage") {
                                        if let Some(p) = usage
                                            .get("prompt_tokens")
                                            .and_then(|v| v.as_u64())
                                        {
                                            exact_prompt_tokens = p;
                                        }
                                        if let Some(c) = usage
                                            .get("completion_tokens")
                                            .and_then(|v| v.as_u64())
                                        {
                                            exact_completion_tokens = c;
                                        }
                                    }
                                }
                            }

                            let sse = format!("{}\n\n", line);
                            if tx.send(Ok(Bytes::from(sse))).await.is_err() {
                                return; // client disconnected
                            }
                        }
                    }
                }
            }

            let duration = start.elapsed().as_secs_f64();
            let total_tokens = exact_prompt_tokens + exact_completion_tokens;

            if exact_completion_tokens > 0 {
                state
                    .metrics
                    .tokens_generated
                    .get_or_create(&ModelLabel { model: model.clone() })
                    .inc_by(exact_completion_tokens);
            }
            if exact_prompt_tokens > 0 {
                state
                    .metrics
                    .tokens_prompted
                    .get_or_create(&ModelLabel { model: model.clone() })
                    .inc_by(exact_prompt_tokens);
            }

            state
                .metrics
                .request_count
                .get_or_create(&RequestLabels {
                    method: "POST".into(),
                    endpoint: path_str.clone(),
                    status_code: status.as_str().to_string(),
                    model: model.clone(),
                })
                .inc_by(1);

            state
                .metrics
                .request_latency
                .get_or_create(&EndpointModelLabels {
                    endpoint: path_str,
                    model: model.clone(),
                })
                .observe(duration);

            info!(
                duration_ms = (duration * 1000.0) as u64,
                prompt_tokens = exact_prompt_tokens,
                completion_tokens = exact_completion_tokens,
                tok_per_sec = if duration > 0.0 {
                    (total_tokens as f64 / duration) as u64
                } else {
                    0
                },
                ttft_ms = (ttft * 1000.0) as u64,
                backend = %backend.url,
                "stream complete"
            );

            if total_tokens > 0 {
                state
                    .quota
                    .add_tokens(&api_key, exact_prompt_tokens, exact_completion_tokens);
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .header("connection", "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap())
}

// ── Embeddings ───────────────────────────────────────────────────────────────

/// POST /v1/embeddings — auth + rate limit + quota; proxies to the backend that
/// serves the requested model (no model-name rewrite).
pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ApiKey(api_key): ApiKey,
    body: Bytes,
) -> Result<Response, GatewayError> {
    proxy_post_simple(state, api_key, addr.ip(), body, "/v1/embeddings").await
}

/// Non-streaming proxy for endpoints that never stream (embeddings).
/// Does NOT rewrite model name so a dedicated embedding model can be used.
async fn proxy_post_simple(
    state: Arc<AppState>,
    api_key: String,
    client_ip: std::net::IpAddr,
    raw_body: Bytes,
    path: &'static str,
) -> Result<Response, GatewayError> {
    if state.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::ShuttingDown);
    }
    if !state.rate_limiters.check(client_ip) {
        state.metrics.rate_limited.inc();
        return Err(GatewayError::RateLimited);
    }

    // Per-key RPM + daily limit
    let key_hash = sha256_hex(&api_key);
    let key_meta = state.key_store.lookup(&key_hash);
    if let Some(ref meta) = key_meta {
        if let Some(rpm) = meta.rpm_limit {
            if !state.rate_limiters.check_key(&key_hash, rpm as u32) {
                state.metrics.rate_limited.inc();
                return Err(GatewayError::RateLimited);
            }
        }
    }
    let per_key_limit = key_meta.as_ref().and_then(|m| m.daily_token_limit.map(|l| l as u64));
    if !state.quota.is_allowed(&api_key, per_key_limit) {
        state.metrics.quota_exceeded.inc();
        return Err(GatewayError::QuotaExceeded);
    }

    let body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| GatewayError::BadRequest(e.to_string()))?;
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.config.served_model_name)
        .to_string();

    // Select backend — embeddings endpoints route by model name, no rewrite
    let backend = state
        .backend_router
        .select(&model)
        .ok_or_else(|| GatewayError::UpstreamError(
            format!("no available backend for model '{}'", model),
        ))?;

    backend.circuit_breaker.before_call()?;
    let start = Instant::now();

    let upstream_url = format!("{}{}", backend.url, path);
    let resp = state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", backend.api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            backend.circuit_breaker.on_failure();
            if e.is_timeout() {
                GatewayError::UpstreamTimeout
            } else {
                GatewayError::UpstreamError(e.to_string())
            }
        })?;

    let status = resp.status();
    let duration = start.elapsed().as_secs_f64();

    if status.is_server_error() {
        backend.circuit_breaker.on_failure();
    } else {
        backend.circuit_breaker.on_success();
    }

    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    let prompt_tokens = serde_json::from_slice::<Value>(&body_bytes)
        .ok()
        .and_then(|v| v.pointer("/usage/prompt_tokens").and_then(|t| t.as_u64()))
        .unwrap_or(0);

    if prompt_tokens > 0 {
        state.quota.add_tokens(&api_key, prompt_tokens, 0);
        state
            .metrics
            .tokens_prompted
            .get_or_create(&ModelLabel { model: model.clone() })
            .inc_by(prompt_tokens);
    }

    state
        .metrics
        .request_count
        .get_or_create(&RequestLabels {
            method: "POST".into(),
            endpoint: path.to_string(),
            status_code: status.as_str().to_string(),
            model: model.clone(),
        })
        .inc_by(1);

    state
        .metrics
        .request_latency
        .get_or_create(&EndpointModelLabels {
            endpoint: path.to_string(),
            model,
        })
        .observe(duration);

    info!(
        status = status.as_u16(),
        duration_ms = (duration * 1000.0) as u64,
        prompt_tokens,
        "request complete"
    );

    let axum_status =
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    Ok(Response::builder()
        .status(axum_status)
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap())
}

// ── Tokenize / Detokenize ─────────────────────────────────────────────────────

/// POST /v1/tokenize — auth-gated passthrough to primary backend /tokenize (no quota).
pub async fn tokenize_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ApiKey(_api_key): ApiKey,
    body: Bytes,
) -> Result<Response, GatewayError> {
    passthrough_post(state, addr.ip(), body, "/tokenize").await
}

/// POST /v1/detokenize — auth-gated passthrough to primary backend /detokenize (no quota).
pub async fn detokenize_handler(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    ApiKey(_api_key): ApiKey,
    body: Bytes,
) -> Result<Response, GatewayError> {
    passthrough_post(state, addr.ip(), body, "/detokenize").await
}

/// Auth + rate-limit passthrough to the primary backend — rewrites model name, no quota.
async fn passthrough_post(
    state: Arc<AppState>,
    client_ip: std::net::IpAddr,
    raw_body: Bytes,
    vllm_path: &'static str,
) -> Result<Response, GatewayError> {
    if state.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::ShuttingDown);
    }
    if !state.rate_limiters.check(client_ip) {
        state.metrics.rate_limited.inc();
        return Err(GatewayError::RateLimited);
    }

    let backend = state
        .backend_router
        .primary()
        .ok_or_else(|| GatewayError::UpstreamError("no backends configured".into()))?;

    // Rewrite model to the backend's served name so vLLM uses the right tokenizer.
    let mut body: Value = serde_json::from_slice(&raw_body)
        .map_err(|e| GatewayError::BadRequest(e.to_string()))?;
    if let Some(ref served_name) = backend.served_model_name {
        body["model"] = Value::String(served_name.clone());
    }

    let upstream_url = format!("{}{}", backend.url, vllm_path);
    let resp = state
        .client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", backend.api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                GatewayError::UpstreamTimeout
            } else {
                GatewayError::UpstreamError(e.to_string())
            }
        })?;

    let status_code =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    Ok(Response::builder()
        .status(status_code)
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap())
}
