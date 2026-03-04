"""
OpenAI SDK compatibility tests for the LLM gateway.

All tests use the OpenAI Python SDK against the FastAPI gateway running in-process
(httpx ASGITransport).  vLLM is mocked via respx — no GPU or vLLM process is needed.

Model: TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF:Q8_0
"""
from pathlib import Path

import pytest
import httpx
import respx
from openai import AsyncOpenAI, AuthenticationError, APIStatusError

from tests.conftest import MODEL, CHAT_RESPONSE, GATEWAY_BASE, _sse_stream

pytestmark = pytest.mark.asyncio


# ── /health (no auth) ─────────────────────────────────────────────────────────

async def test_health(openai_client: AsyncOpenAI):
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /v1/models ────────────────────────────────────────────────────────────────

async def test_list_models(vllm_mock, openai_client: AsyncOpenAI):
    models = await openai_client.models.list()
    ids = [m.id for m in models.data]
    assert MODEL in ids


async def test_list_models_bad_key(openai_client_bad_key: AsyncOpenAI):
    with pytest.raises(AuthenticationError) as exc_info:
        await openai_client_bad_key.models.list()
    assert "Invalid or missing API key" in str(exc_info.value)


# ── /v1/chat/completions — non-streaming ─────────────────────────────────────

async def test_chat_completion(vllm_mock, openai_client: AsyncOpenAI):
    response = await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello"}],
    )
    assert response.choices[0].message.content == "Hello! How can I help?"
    assert response.usage.total_tokens == 18


async def test_chat_completion_bad_key(vllm_mock, openai_client_bad_key: AsyncOpenAI):
    with pytest.raises(AuthenticationError):
        await openai_client_bad_key.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
        )


async def test_chat_completion_invalid_json(openai_client: AsyncOpenAI):
    """Gateway must return 400 invalid_request_error for malformed JSON bodies."""
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json", "authorization": "Bearer test-key-1"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"


# ── /v1/chat/completions — streaming ─────────────────────────────────────────

async def test_chat_completion_stream(openai_client: AsyncOpenAI):
    """SSE streaming must be parseable token-by-token by the OpenAI SDK."""
    sse_bytes = _sse_stream("Hello world from the model")

    with respx.mock(base_url="http://fake-vllm:8000", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").respond(
            200,
            content=sse_bytes,
            headers={"content-type": "text/event-stream"},
        )

        collected = []
        async with await openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)

    assert len(collected) > 0
    assert "Hello" in "".join(collected)


async def test_chat_completion_stream_bad_key(openai_client_bad_key: AsyncOpenAI):
    with pytest.raises(AuthenticationError):
        async with await openai_client_bad_key.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ) as stream:
            async for _ in stream:
                pass


# ── Error response format ─────────────────────────────────────────────────────

async def test_error_format_401(openai_client_bad_key: AsyncOpenAI):
    """401 must use OpenAI-format: {"error": {"message", "type", "param", "code"}}."""
    raw: httpx.AsyncClient = openai_client_bad_key._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert set(body["error"].keys()) == {"message", "type", "param", "code"}
    assert body["error"]["type"] == "authentication_error"
    assert body["error"]["param"] is None
    assert body["error"]["code"] is None


async def test_error_format_upstream_503(openai_client: AsyncOpenAI):
    """
    Upstream 5xx: gateway returns the upstream status code and body is JSON.
    The gateway passes through the upstream error body unchanged — the OpenAI
    SDK will raise APIStatusError regardless as long as status >= 400.
    """
    with respx.mock(base_url="http://fake-vllm:8000", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").respond(503, json={"error": "vllm down"})

        raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
        resp = await raw.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer test-key-1"},
        )

    assert resp.status_code in (503, 502, 500)
    # Gateway passes through upstream error body as-is — it must at least be JSON
    body = resp.json()
    assert isinstance(body, dict)


# ── /v1/usage (custom gateway endpoint) ──────────────────────────────────────

async def test_usage_endpoint(openai_client: AsyncOpenAI):
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/usage", headers={"authorization": "Bearer test-key-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert "prompt_tokens" in body
    assert "quota_remaining" in body


# ── GET /v1/models/local ──────────────────────────────────────────────────────

async def test_local_models_requires_auth(openai_client_bad_key: AsyncOpenAI):
    """Endpoint must reject unauthenticated callers."""
    raw: httpx.AsyncClient = openai_client_bad_key._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models/local")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["type"] == "authentication_error"


async def test_local_models_empty_cache(openai_client: AsyncOpenAI, tmp_path: Path, monkeypatch):
    """Empty MODEL_CACHE_DIR returns an empty list, not an error."""
    monkeypatch.setattr("gateway.app.settings.model_cache_dir", str(tmp_path))
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models/local", headers={"authorization": "Bearer test-key-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []


async def test_local_models_missing_dir(openai_client: AsyncOpenAI, monkeypatch):
    """Non-existent MODEL_CACHE_DIR returns an empty list gracefully."""
    monkeypatch.setattr("gateway.app.settings.model_cache_dir", "/nonexistent/path")
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models/local", headers={"authorization": "Bearer test-key-1"})
    assert resp.status_code == 200
    assert resp.json()["data"] == []


async def test_local_models_finds_gguf(openai_client: AsyncOpenAI, model_cache: Path, monkeypatch):
    """GGUF files are discovered and returned with relative paths."""
    monkeypatch.setattr("gateway.app.settings.model_cache_dir", str(model_cache))
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models/local", headers={"authorization": "Bearer test-key-1"})
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    # Both GGUF files should appear (paths relative to cache_dir)
    assert any("Q8_0.gguf" in i for i in ids)
    assert any("Q4_K_M.gguf" in i for i in ids)
    # All GGUF entries have format="gguf"
    gguf_entries = [m for m in resp.json()["data"] if m.get("format") == "gguf"]
    assert len(gguf_entries) == 2


async def test_local_models_finds_hf_dir(openai_client: AsyncOpenAI, model_cache: Path, monkeypatch):
    """HuggingFace cache dirs are converted to org/repo IDs."""
    monkeypatch.setattr("gateway.app.settings.model_cache_dir", str(model_cache))
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models/local", headers={"authorization": "Bearer test-key-1"})
    ids = [m["id"] for m in resp.json()["data"]]
    assert "meta-llama/Llama-3.2-7B-Instruct" in ids


async def test_local_models_openai_schema(openai_client: AsyncOpenAI, model_cache: Path, monkeypatch):
    """Each entry must have all OpenAI Model-list fields so clients can copy-paste."""
    monkeypatch.setattr("gateway.app.settings.model_cache_dir", str(model_cache))
    raw: httpx.AsyncClient = openai_client._client  # type: ignore[attr-defined]
    resp = await raw.get("/v1/models/local", headers={"authorization": "Bearer test-key-1"})
    for entry in resp.json()["data"]:
        assert "id" in entry
        assert entry["object"] == "model"
        assert "owned_by" in entry
        assert "format" in entry  # extra field: "gguf" or "hf"
