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
  - OpenTelemetry distributed tracing → Jaeger
  - Circuit breaker for vLLM upstream
  - Per-key daily token quota via SQLite
  - Graceful shutdown with in-flight request draining
"""

import asyncio
import json
import os
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from gateway import metrics
from gateway.auth import require_api_key
from gateway.circuit_breaker import CircuitBreaker, CircuitOpenError
from gateway.config import settings
from gateway.logging_config import configure_logging
from gateway.usage_db import check_quota, get_usage, init_db, record_usage

configure_logging()
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------
_tracer = None

def _setup_otel():
    global _tracer
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "llm-gateway"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("llm-gateway")
        log.info("otel_initialized", endpoint=settings.otel_exporter_otlp_endpoint)
    except Exception as exc:
        log.warning("otel_init_failed", error=str(exc))


def get_tracer():
    return _tracer


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[])

# ---------------------------------------------------------------------------
# Circuit breaker (module-level singleton)
# ---------------------------------------------------------------------------
circuit_breaker = CircuitBreaker(
    failure_threshold=settings.cb_failure_threshold,
    recovery_timeout=settings.cb_recovery_timeout,
    half_open_max_calls=settings.cb_half_open_max_calls,
)

# ---------------------------------------------------------------------------
# Shutdown state
# ---------------------------------------------------------------------------
_shutting_down = asyncio.Event()
# WeakSet of asyncio Tasks currently processing requests
_inflight_tasks: weakref.WeakSet = weakref.WeakSet()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_otel()
    await init_db()

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

    # Drain in-flight requests
    tasks = list(_inflight_tasks)
    if tasks:
        log.info("gateway_draining", inflight=len(tasks))
        done, pending = await asyncio.wait(tasks, timeout=30)
        if pending:
            log.warning("gateway_drain_timeout", abandoned=len(pending))
        log.info("gateway_drained", completed=len(done), abandoned=len(pending) if pending else 0)

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


def _openai_error(message: str, error_type: str, status_code: int, headers=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
        headers=headers,
    )


def _status_to_type(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 422 or status_code == 400:
        return "invalid_request_error"
    if status_code == 429:
        return "rate_limit_error"
    return "api_error"


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _openai_error(message, _status_to_type(exc.status_code), exc.status_code, exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _openai_error(str(exc), "invalid_request_error", 422)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    metrics.RATE_LIMITED.inc()
    return _openai_error("Rate limit exceeded", "rate_limit_error", 429)


# ---------------------------------------------------------------------------
# Middleware: reject new requests during shutdown
# ---------------------------------------------------------------------------
@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    if _shutting_down.is_set() and request.url.path not in ("/health", "/metrics"):
        return Response(
            content='{"detail":"Service shutting down"}',
            status_code=503,
            media_type="application/json",
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Middleware: request-id injection + OTel context binding
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
# Usage endpoint — per-key daily token usage
# ---------------------------------------------------------------------------
@app.get("/v1/usage")
async def usage_endpoint(
    request: Request,
    api_key: str = Depends(require_api_key),
):
    quota_info = await check_quota(api_key)
    usage = await get_usage(api_key)
    return {
        "date": usage["date"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "requests": usage["request_count"],
        "quota": quota_info["quota"],
        "quota_remaining": quota_info["remaining"],
    }


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
# Local models — list weight files available on disk (auth required)
# ---------------------------------------------------------------------------
def _scan_local_models(cache_dir: str) -> list[dict]:
    """
    Scan MODEL_CACHE_DIR and return a list of model descriptors.

    Recognises two layouts:
      • GGUF files  — any *.gguf file found recursively
      • HF snapshot dirs — directories named "models--<org>--<repo>"
        (the standard HuggingFace Hub cache layout)

    Returns dicts compatible with the OpenAI /v1/models "Model" schema so
    clients can copy-paste the "id" straight into their requests.
    """
    root = Path(cache_dir)
    results: list[dict] = []

    if not root.exists():
        return results

    # GGUF files — return path relative to cache_dir so clients see e.g.
    # "TeichAI/model.Q8_0.gguf" rather than an absolute machine path.
    for gguf in sorted(root.rglob("*.gguf")):
        try:
            rel = str(gguf.relative_to(root))
        except ValueError:
            rel = gguf.name
        results.append({"id": rel, "object": "model", "created": 0, "owned_by": "local", "format": "gguf"})

    # HuggingFace Hub cache dirs — convert "models--org--repo" → "org/repo"
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith("models--"):
            parts = entry.name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                hf_id = f"{parts[0]}/{parts[1]}"
                results.append({"id": hf_id, "object": "model", "created": 0, "owned_by": "local", "format": "hf"})

    return results


@app.get("/v1/models/local")
async def list_local_models(
    _api_key: str = Depends(require_api_key),
):
    """List model weight files available on disk — no vLLM call needed."""
    models = _scan_local_models(settings.model_cache_dir)
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Core proxy logic
# ---------------------------------------------------------------------------
async def _proxy_request(request: Request, upstream_path: str) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    model = settings.served_model_name
    endpoint = upstream_path
    start = time.monotonic()
    metrics.ACTIVE_REQUESTS.inc()

    # Track this task for graceful drain
    current_task = asyncio.current_task()
    if current_task:
        _inflight_tasks.add(current_task)
        metrics.INFLIGHT_TASKS.set(len(list(_inflight_tasks)))

    tracer = get_tracer()

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

        # --- Quota check ---
        api_key = request.headers.get("x-api-key") or ""
        if not api_key:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        quota_info = await check_quota(api_key)
        if not quota_info["allowed"]:
            metrics.QUOTA_EXCEEDED.inc()
            return Response(
                content=json.dumps({"detail": "Daily token quota exceeded"}),
                status_code=429,
                media_type="application/json",
                headers={"X-Quota-Remaining": "0"},
            )

        log.info(
            "request_received",
            endpoint=endpoint,
            model=model,
            stream=stream,
            messages=len(body.get("messages", [])),
            max_tokens=body.get("max_tokens"),
        )

        # --- OTel span ---
        if tracer:
            from opentelemetry import trace
            from opentelemetry.propagate import inject

            with tracer.start_as_current_span(
                "llm.request",
                kind=trace.SpanKind.SERVER,
            ) as span:
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.stream", stream)
                span.set_attribute("llm.endpoint", endpoint)
                if body.get("max_tokens"):
                    span.set_attribute("llm.max_tokens", body["max_tokens"])

                # Propagate trace context downstream to vLLM
                carrier: dict = {}
                inject(carrier)
                if "traceparent" in carrier:
                    body["_traceparent"] = carrier["traceparent"]

                if stream:
                    return await _stream_response(client, endpoint, body, model, start, api_key, span)
                else:
                    return await _sync_response(client, endpoint, body, model, start, api_key, span)
        else:
            if stream:
                return await _stream_response(client, endpoint, body, model, start, api_key, None)
            else:
                return await _sync_response(client, endpoint, body, model, start, api_key, None)

    except HTTPException:
        raise
    except CircuitOpenError as exc:
        duration = time.monotonic() - start
        metrics.REQUEST_COUNT.labels("POST", endpoint, "503", model).inc()
        log.warning("circuit_open", duration=round(duration, 3))
        raise HTTPException(status_code=503, detail=str(exc))
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
        if current_task:
            metrics.INFLIGHT_TASKS.set(len(list(_inflight_tasks)))


async def _sync_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    model: str,
    start: float,
    api_key: str,
    span=None,
) -> Response:
    upstream_headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}

    # Propagate traceparent if present
    if "_traceparent" in body:
        upstream_headers["traceparent"] = body.pop("_traceparent")
    else:
        body.pop("_traceparent", None)

    tracer = get_tracer()
    prompt_tokens = 0
    completion_tokens = 0

    async def _do_request():
        nonlocal prompt_tokens, completion_tokens
        await circuit_breaker.before_call()
        try:
            resp = await client.post(path, json=body, headers=upstream_headers)
            duration = time.monotonic() - start

            if resp.status_code >= 500:
                await circuit_breaker.on_failure()
                metrics.UPSTREAM_ERRORS.labels(str(resp.status_code)).inc()
                log.warning("upstream_error", status=resp.status_code, body=resp.text[:500])
            else:
                await circuit_breaker.on_success()

            # Extract token counts
            try:
                resp_body = resp.json()
                usage = resp_body.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                if completion_tokens:
                    metrics.TOKENS_GENERATED.labels(model).inc(completion_tokens)
                if prompt_tokens:
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
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except (httpx.TimeoutException, httpx.RequestError):
            await circuit_breaker.on_failure()
            raise

    if tracer and span:
        from opentelemetry import trace
        with tracer.start_as_current_span("llm.upstream_call", kind=trace.SpanKind.CLIENT) as child:
            child.set_attribute("http.method", "POST")
            child.set_attribute("http.url", path)
            result = await _do_request()
            child.set_attribute("http.status_code", result.status_code)
    else:
        result = await _do_request()

    # Record usage after successful response
    if api_key and (prompt_tokens or completion_tokens):
        await record_usage(api_key, prompt_tokens, completion_tokens)

    if span:
        span.set_attribute("llm.prompt_tokens", prompt_tokens)
        span.set_attribute("llm.completion_tokens", completion_tokens)

    return result


async def _stream_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    model: str,
    start: float,
    api_key: str,
    span=None,
) -> StreamingResponse:
    tracer = get_tracer()

    # Extract traceparent before streaming
    upstream_headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}
    if "_traceparent" in body:
        upstream_headers["traceparent"] = body.pop("_traceparent")
    else:
        body.pop("_traceparent", None)

    async def generator() -> AsyncIterator[bytes]:
        first_token = True
        token_count = 0
        status_code = 200
        ttft = 0.0

        try:
            await circuit_breaker.before_call()
            upstream_ok = False

            try:
                async with client.stream(
                    "POST",
                    path,
                    json={**body, "stream": True},
                    headers=upstream_headers,
                ) as resp:
                    status_code = resp.status_code

                    if resp.status_code >= 500:
                        await circuit_breaker.on_failure()
                        metrics.UPSTREAM_ERRORS.labels(str(resp.status_code)).inc()
                        error_body = await resp.aread()
                        log.error("upstream_stream_error", status=resp.status_code)
                        yield error_body
                        return
                    else:
                        upstream_ok = True

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

            except (httpx.TimeoutException, httpx.RequestError):
                await circuit_breaker.on_failure()
                raise

            if upstream_ok:
                await circuit_breaker.on_success()

        except CircuitOpenError:
            log.warning("circuit_open_stream")
            yield b'data: {"error": "upstream circuit breaker open"}\n\n'
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
                ttft=round(ttft, 3),
            )

            # Record usage
            if api_key and token_count:
                try:
                    await record_usage(api_key, 0, token_count)
                except Exception:
                    pass

            if span:
                span.set_attribute("llm.approx_completion_tokens", token_count)
                span.set_attribute("llm.ttft_seconds", round(ttft, 3))

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tells nginx not to buffer SSE
            "Connection": "keep-alive",
        },
    )
