"""
Production API gateway for vLLM.

Responsibilities:
  - TLS termination is handled upstream (nginx/caddy); we speak plain HTTP
  - Authentication via Bearer token or x-api-key header
  - Per-IP rate limiting (slowapi / in-memory token bucket)
  - Request/response logging with structured JSON
  - Prometheus metrics (latency histograms, TTFT, token counters)
  - Transparent proxy to vLLM's OpenAI-compatible API
  - Streaming (SSE) passthrough with TTFT measurement
  - Health + readiness endpoints
  - /metrics endpoint (restrict to internal network in prod via nginx)
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from gateway import metrics
from gateway.auth import require_api_key
from gateway.config import settings
from gateway.logging_config import configure_logging

configure_logging()
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[])


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("gateway_starting", vllm_url=settings.vllm_url)
    app.state.http_client = httpx.AsyncClient(
        base_url=settings.vllm_url,
        timeout=httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.request_timeout_seconds,
            write=30.0,
            pool=5.0,
        ),
        limits=httpx.Limits(
            max_connections=500,
            max_keepalive_connections=100,
            keepalive_expiry=30,
        ),
        http2=True,
    )
    app.state.warmed_up = False

    # Warmup: wait for vLLM to become healthy before marking ready
    asyncio.create_task(_warmup(app))

    yield

    await app.state.http_client.aclose()
    log.info("gateway_stopped")


async def _warmup(app: FastAPI) -> None:
    """Poll vLLM health until it responds, then mark gateway ready."""
    client: httpx.AsyncClient = app.state.http_client
    for attempt in range(60):  # up to 5 minutes
        try:
            resp = await client.get("/health", timeout=5.0)
            if resp.status_code == 200:
                app.state.warmed_up = True
                log.info("gateway_ready", attempts=attempt + 1)
                return
        except Exception:
            pass
        await asyncio.sleep(5)
    log.warning("warmup_timed_out")
    app.state.warmed_up = True  # unblock anyway — let requests fail naturally


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Gateway",
    version="0.1.0",
    docs_url=None,   # disable swagger in prod
    redoc_url=None,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Middleware: request-id injection
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        client_ip=request.client.host if request.client else "unknown",
    )
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Middleware: record rate-limited requests
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 429:
        metrics.RATE_LIMITED.inc()
    return response


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def ready(request: Request):
    if not request.app.state.warmed_up:
        raise HTTPException(status_code=503, detail="Warming up — vLLM not yet healthy")
    try:
        resp = await request.app.state.http_client.get("/health", timeout=3.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=503, detail="vLLM upstream unhealthy")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach vLLM: {exc}")
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Chat completions  (OpenAI-compatible)
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_completions(
    request: Request,
    _api_key: str = Depends(require_api_key),
):
    return await _proxy_request(request, "/v1/chat/completions")


# ---------------------------------------------------------------------------
# Completions  (legacy)
# ---------------------------------------------------------------------------
@app.post("/v1/completions")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def completions(
    request: Request,
    _api_key: str = Depends(require_api_key),
):
    return await _proxy_request(request, "/v1/completions")


# ---------------------------------------------------------------------------
# Models list — useful for clients that auto-detect available models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models(
    request: Request,
    _api_key: str = Depends(require_api_key),
):
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            timeout=10.0,
        )
        return Response(content=resp.content, media_type="application/json", status_code=resp.status_code)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# Core proxy logic
# ---------------------------------------------------------------------------
async def _proxy_request(request: Request, upstream_path: str) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    model = settings.served_model_name
    endpoint = upstream_path
    start = time.monotonic()
    metrics.ACTIVE_REQUESTS.inc()

    try:
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON")

        stream = body.get("stream", False)
        model = body.get("model", settings.served_model_name)

        # Rewrite model name to whatever vLLM is serving
        body["model"] = settings.served_model_name

        log.info(
            "request_received",
            endpoint=endpoint,
            model=model,
            stream=stream,
            messages=len(body.get("messages", [])),
            max_tokens=body.get("max_tokens"),
        )

        if stream:
            return await _stream_response(client, endpoint, body, model, start)
        else:
            return await _sync_response(client, endpoint, body, model, start)

    except HTTPException:
        raise
    except httpx.TimeoutException:
        duration = time.monotonic() - start
        metrics.REQUEST_COUNT.labels("POST", endpoint, "504", model).inc()
        metrics.REQUEST_LATENCY.labels(endpoint, model).observe(duration)
        log.error("upstream_timeout", duration=round(duration, 3))
        raise HTTPException(status_code=504, detail="Upstream LLM request timed out")
    except httpx.RequestError as exc:
        duration = time.monotonic() - start
        metrics.REQUEST_COUNT.labels("POST", endpoint, "503", model).inc()
        metrics.REQUEST_LATENCY.labels(endpoint, model).observe(duration)
        log.error("upstream_connection_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Cannot reach LLM backend: {exc}")
    except Exception:
        duration = time.monotonic() - start
        metrics.REQUEST_COUNT.labels("POST", endpoint, "500", model).inc()
        metrics.REQUEST_LATENCY.labels(endpoint, model).observe(duration)
        log.exception("unhandled_error")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        metrics.ACTIVE_REQUESTS.dec()


async def _sync_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    model: str,
    start: float,
) -> Response:
    resp = await client.post(
        path,
        json=body,
        headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
    )
    duration = time.monotonic() - start

    if resp.status_code != 200:
        metrics.UPSTREAM_ERRORS.labels(str(resp.status_code)).inc()
        log.warning("upstream_error", status=resp.status_code, body=resp.text[:500])

    # Extract token counts from response for metrics
    try:
        resp_body = resp.json()
        usage = resp_body.get("usage", {})
        if completion_tokens := usage.get("completion_tokens"):
            metrics.TOKENS_GENERATED.labels(model).inc(completion_tokens)
        if prompt_tokens := usage.get("prompt_tokens"):
            metrics.TOKENS_PROMPTED.labels(model).inc(prompt_tokens)
    except Exception:
        pass

    metrics.REQUEST_COUNT.labels("POST", path, str(resp.status_code), model).inc()
    metrics.REQUEST_LATENCY.labels(path, model).observe(duration)
    log.info(
        "request_complete",
        status=resp.status_code,
        duration=round(duration, 3),
        stream=False,
    )
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


async def _stream_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    model: str,
    start: float,
) -> StreamingResponse:

    async def generator() -> AsyncIterator[bytes]:
        first_token = True
        token_count = 0
        status_code = 200

        try:
            async with client.stream(
                "POST",
                path,
                json={**body, "stream": True},
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            ) as resp:
                status_code = resp.status_code

                if resp.status_code != 200:
                    metrics.UPSTREAM_ERRORS.labels(str(resp.status_code)).inc()
                    error_body = await resp.aread()
                    log.error("upstream_stream_error", status=resp.status_code)
                    yield error_body
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue

                    yield (raw_line + "\n\n").encode()

                    if raw_line.startswith("data: ") and raw_line != "data: [DONE]":
                        if first_token:
                            ttft = time.monotonic() - start
                            metrics.TTFT.labels(model).observe(ttft)
                            log.info("first_token", ttft=round(ttft, 3))
                            first_token = False

                        try:
                            chunk = json.loads(raw_line[6:])
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                token_count += len(delta.split())  # rough approximation
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

        except httpx.TimeoutException:
            log.error("stream_timeout")
            yield b'data: {"error": "upstream timeout"}\n\n'
        except Exception:
            log.exception("stream_error")
        finally:
            duration = time.monotonic() - start
            metrics.TOKENS_GENERATED.labels(model).inc(token_count)
            metrics.REQUEST_COUNT.labels("POST", path, str(status_code), model).inc()
            metrics.REQUEST_LATENCY.labels(path, model).observe(duration)
            log.info(
                "stream_complete",
                duration=round(duration, 3),
                approx_tokens=token_count,
                tok_per_sec=round(token_count / duration, 1) if duration > 0 else 0,
            )

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tells nginx not to buffer SSE
            "Connection": "keep-alive",
        },
    )
