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
use std::{net::SocketAddr, sync::Arc, time::Instant};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

use crate::{
    auth::ApiKey,
    error::GatewayError,
    metrics::{EndpointModelLabels, ModelLabel, RequestLabels, StatusCodeLabel},
    AppState,
};

// ── Public route handlers ────────────────────────────────────────────────────

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

    let quota = state.config.daily_token_quota;
    let used = (info.prompt_tokens + info.completion_tokens) as u64;
    let remaining: i64 = if quota == 0 {
        -1
    } else {
        (quota as i64 - used as i64).max(0)
    };

    Ok(Json(serde_json::json!({
        "date":               info.date,
        "prompt_tokens":      info.prompt_tokens,
        "completion_tokens":  info.completion_tokens,
        "requests":           info.request_count,
        "quota":              quota,
        "quota_remaining":    remaining,
    })))
}

/// GET /v1/models — passthrough to vLLM
pub async fn models_handler(
    State(state): State<Arc<AppState>>,
    ApiKey(_api_key): ApiKey,
) -> Result<impl IntoResponse, GatewayError> {
    if state.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(GatewayError::ShuttingDown);
    }

    let resp = state
        .client
        .get(format!("{}/v1/models", state.config.vllm_url))
        .header(
            "Authorization",
            format!("Bearer {}", state.config.vllm_api_key),
        )
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    let status = StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| GatewayError::UpstreamError(e.to_string()))?;

    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap())
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

    // ── Rate limit (per-IP GCRA) ─────────────────────────────────────────────
    if !state.rate_limiters.check(client_ip) {
        state.metrics.rate_limited.inc();
        return Err(GatewayError::RateLimited);
    }

    // ── Daily quota check ────────────────────────────────────────────────────
    if !state.quota.is_allowed(&api_key) {
        state.metrics.quota_exceeded.inc();
        return Err(GatewayError::QuotaExceeded);
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

    // Rewrite model name to whatever vLLM is actually serving
    body["model"] = Value::String(state.config.served_model_name.clone());

    state.metrics.active_requests.inc();
    state.metrics.inflight_requests.inc();

    let result = if stream {
        stream_proxy(state.clone(), api_key, body, &model, path).await
    } else {
        sync_proxy(state.clone(), api_key, body, &model, path).await
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
) -> Result<Response, GatewayError> {
    let start = Instant::now();

    state.circuit_breaker.before_call()?;

    let upstream_url = format!("{}{}", state.config.vllm_url, path);

    let resp = state
        .client
        .post(&upstream_url)
        .header(
            "Authorization",
            format!("Bearer {}", state.config.vllm_api_key),
        )
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            state.circuit_breaker.on_failure();
            if e.is_timeout() {
                GatewayError::UpstreamTimeout
            } else {
                GatewayError::UpstreamError(e.to_string())
            }
        })?;

    let status = resp.status();
    let duration = start.elapsed().as_secs_f64();

    if status.is_server_error() {
        state.circuit_breaker.on_failure();
        state
            .metrics
            .upstream_errors
            .get_or_create(&StatusCodeLabel {
                status_code: status.as_str().to_string(),
            })
            .inc_by(1);
        warn!(status = status.as_u16(), "upstream error");
    } else {
        state.circuit_breaker.on_success();
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
            .get_or_create(&ModelLabel {
                model: model.to_string(),
            })
            .inc_by(prompt_tokens);
    }
    if completion_tokens > 0 {
        state
            .metrics
            .tokens_generated
            .get_or_create(&ModelLabel {
                model: model.to_string(),
            })
            .inc_by(completion_tokens);
    }

    info!(
        status = status.as_u16(),
        duration_ms = (duration * 1000.0) as u64,
        prompt_tokens,
        completion_tokens,
        "request complete"
    );

    if prompt_tokens > 0 || completion_tokens > 0 {
        state
            .quota
            .add_tokens(&api_key, prompt_tokens, completion_tokens);
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
) -> Result<Response, GatewayError> {
    let start = Instant::now();
    body["stream"] = Value::Bool(true);

    // Check circuit breaker before spawning the background task.
    state.circuit_breaker.before_call()?;

    let upstream_url = format!("{}{}", state.config.vllm_url, path);
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
                .header(
                    "Authorization",
                    format!("Bearer {}", state.config.vllm_api_key),
                )
                .json(&body)
                .send()
                .await;

            let resp = match resp_result {
                Ok(r) => r,
                Err(e) => {
                    state.circuit_breaker.on_failure();
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
                state.circuit_breaker.on_failure();
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

            state.circuit_breaker.on_success();

            let mut byte_stream = resp.bytes_stream();
            let mut line_buf: Vec<u8> = Vec::new();
            let mut first_token = true;
            let mut approx_tokens: u64 = 0;
            let mut ttft = 0f64;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        error!(error = %e, "stream chunk error");
                        break;
                    }
                };

                line_buf.extend_from_slice(&chunk);

                // Drain complete `\n`-terminated lines from the buffer
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

                            // Measure TTFT on first content data line
                            if first_token
                                && line.starts_with("data: ")
                                && line != "data: [DONE]"
                            {
                                ttft = start.elapsed().as_secs_f64();
                                state
                                    .metrics
                                    .ttft
                                    .get_or_create(&ModelLabel {
                                        model: model.clone(),
                                    })
                                    .observe(ttft);
                                info!(ttft_ms = (ttft * 1000.0) as u64, "first token");
                                first_token = false;
                            }

                            // Approximate token count from delta content (word split)
                            if line.starts_with("data: ") && line != "data: [DONE]" {
                                if let Ok(chunk_json) =
                                    serde_json::from_str::<Value>(&line[6..])
                                {
                                    if let Some(delta) = chunk_json
                                        .pointer("/choices/0/delta/content")
                                        .and_then(|v| v.as_str())
                                    {
                                        approx_tokens +=
                                            delta.split_whitespace().count() as u64;
                                    }
                                }
                            }

                            // Forward as SSE (double newline separator)
                            let sse = format!("{}\n\n", line);
                            if tx.send(Ok(Bytes::from(sse))).await.is_err() {
                                return; // client disconnected
                            }
                        }
                    }
                }
            }

            let duration = start.elapsed().as_secs_f64();

            if approx_tokens > 0 {
                state
                    .metrics
                    .tokens_generated
                    .get_or_create(&ModelLabel {
                        model: model.clone(),
                    })
                    .inc_by(approx_tokens);
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
                approx_tokens,
                tok_per_sec = if duration > 0.0 {
                    (approx_tokens as f64 / duration) as u64
                } else {
                    0
                },
                ttft_ms = (ttft * 1000.0) as u64,
                "stream complete"
            );

            if approx_tokens > 0 {
                state.quota.add_tokens(&api_key, 0, approx_tokens);
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
