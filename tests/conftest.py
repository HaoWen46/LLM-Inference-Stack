"""
Test fixtures for gateway OpenAI-SDK compatibility tests.

Strategy:
- Set env vars before importing gateway modules (pydantic-settings reads them at import time).
- Patch out the SQLite usage-DB functions with no-op stubs (not under test here).
- Manually wire app.state so no lifespan/warmup runs.
- Use respx to intercept the gateway's outbound httpx calls to the mock vLLM.
- Route the OpenAI SDK through httpx.ASGITransport so no real TCP port is needed.
"""
import os

# ── Must happen before any gateway import ────────────────────────────────────
os.environ.update(
    {
        "VLLM_API_KEY": "vllm-test-secret",
        "VLLM_URL": "http://fake-vllm:8000",
        "GATEWAY_API_KEYS": "test-key-1,test-key-2",
        "SERVED_MODEL_NAME": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF:Q8_0",
        "RATE_LIMIT_PER_MINUTE": "1000",
        "OTEL_ENABLED": "false",
        "DAILY_TOKEN_QUOTA": "0",
    }
)

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from openai import AsyncOpenAI

# Late import so env vars are in place first
from gateway.app import app

MODEL = "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF:Q8_0"
GATEWAY_BASE = "http://test-gateway"

# ── Canned vLLM responses ─────────────────────────────────────────────────────

CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

MODELS_RESPONSE = {
    "object": "list",
    "data": [{"id": MODEL, "object": "model", "created": 1700000000, "owned_by": "vllm"}],
}


def _sse_stream(content: str, model: str = MODEL) -> bytes:
    """Build a minimal SSE byte stream that the OpenAI SDK can parse."""
    words = content.split()
    lines = []
    for i, word in enumerate(words):
        delta_content = word + (" " if i < len(words) - 1 else "")
        chunk = {
            "id": "chatcmpl-stream1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": delta_content}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    stop_chunk = {
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)},
    }
    lines.append(f"data: {json.dumps(stop_chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


# ── Patch usage-DB so we never touch SQLite ───────────────────────────────────

_QUOTA_OK = {"allowed": True, "quota": 0, "used_tokens": 0, "remaining": -1}
_USAGE_EMPTY = {"date": "2026-03-05", "prompt_tokens": 0, "completion_tokens": 0, "request_count": 0}


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    """Replace all SQLite I/O with lightweight stubs for every test."""
    monkeypatch.setattr("gateway.app.init_db", AsyncMock())
    monkeypatch.setattr("gateway.app.check_quota", AsyncMock(return_value=_QUOTA_OK))
    monkeypatch.setattr("gateway.app.record_usage", AsyncMock())
    monkeypatch.setattr("gateway.app.get_usage", AsyncMock(return_value=_USAGE_EMPTY))


# ── Per-test app.state setup ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def _setup_app_state():
    """Wire app.state so handlers work without the real lifespan."""
    client = httpx.AsyncClient(
        base_url="http://fake-vllm:8000",
        timeout=httpx.Timeout(5.0),
    )
    app.state.http_client = client
    app.state.warmed_up = True
    yield
    await client.aclose()


# ── vLLM mock ─────────────────────────────────────────────────────────────────

@pytest.fixture
def vllm_mock():
    """respx router pre-loaded with standard vLLM happy-path responses."""
    with respx.mock(base_url="http://fake-vllm:8000", assert_all_called=False) as mock:
        mock.get("/health").respond(200, json={"status": "ok"})
        mock.get("/v1/models").respond(200, json=MODELS_RESPONSE)
        mock.post("/v1/chat/completions").respond(200, json=CHAT_RESPONSE)
        mock.post("/v1/completions").respond(200, json=CHAT_RESPONSE)
        yield mock


# ── OpenAI client fixtures ────────────────────────────────────────────────────

@pytest.fixture
def openai_client():
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url=GATEWAY_BASE)
    return AsyncOpenAI(
        base_url=f"{GATEWAY_BASE}/v1",
        api_key="test-key-1",
        http_client=http,
        max_retries=0,
    )


@pytest.fixture
def model_cache(tmp_path: Path):
    """
    Populate a temporary MODEL_CACHE_DIR with a realistic layout:
      - GGUF files nested inside an org directory
      - One HuggingFace Hub cache directory (models--org--repo)
    Returns the path so tests can assert against its contents.
    """
    # GGUF files
    gguf_dir = tmp_path / "TeichAI"
    gguf_dir.mkdir()
    (gguf_dir / "GLM-4.7-Flash.Q8_0.gguf").write_bytes(b"")
    (gguf_dir / "GLM-4.7-Flash.Q4_K_M.gguf").write_bytes(b"")

    # HuggingFace Hub cache dir
    hf_dir = tmp_path / "models--meta-llama--Llama-3.2-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")

    return tmp_path


@pytest.fixture
def openai_client_bad_key():
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url=GATEWAY_BASE)
    return AsyncOpenAI(
        base_url=f"{GATEWAY_BASE}/v1",
        api_key="wrong-key",
        http_client=http,
        max_retries=0,
    )
