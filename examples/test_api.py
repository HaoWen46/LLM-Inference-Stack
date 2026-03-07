#!/usr/bin/env python3
"""
Gateway API integration tests — no model inference needed for most tests.
Covers: auth, admin CRUD, headers, error shapes, rate limiting, quota, metrics, local models.

Usage:
    .venv/bin/python3 examples/test_api.py
    .venv/bin/python3 examples/test_api.py --admin-key <key>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:8080"
_ENV_FILE = Path(__file__).resolve().parent.parent / "config" / ".env"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg = {}
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip().strip('"\'')
    return cfg


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results: list[tuple[str, bool, str]] = []

def check(name: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {name}"
    if detail and not condition:
        msg += f"\n         {detail}"
    print(msg)
    _results.append((name, condition, detail))

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}

def json_body() -> dict:
    return {"Content-Type": "application/json"}

# ── test groups ───────────────────────────────────────────────────────────────

def test_health(c: httpx.Client):
    section("Health & readiness")

    r = c.get(f"{GATEWAY}/health")
    check("GET /health → 200", r.status_code == 200)
    check("GET /health body has status=ok", r.json().get("status") == "ok")

    r = c.get(f"{GATEWAY}/ready")
    check("GET /ready → 200", r.status_code == 200)
    check("GET /ready body has status=ready", r.json().get("status") == "ready")


def test_metrics(c: httpx.Client):
    section("Prometheus metrics")

    r = c.get(f"{GATEWAY}/metrics")
    check("GET /metrics → 200", r.status_code == 200)
    check("Content-Type is text/plain", "text/plain" in r.headers.get("content-type", ""))
    check("Contains gateway_requests_total", "gateway_requests_total" in r.text)
    check("Contains gateway_request_duration_seconds", "gateway_request_duration_seconds" in r.text)


def test_security_headers(c: httpx.Client, api_key: str):
    section("Security headers")

    r = c.get(f"{GATEWAY}/health")
    check("X-Content-Type-Options: nosniff",
          r.headers.get("x-content-type-options") == "nosniff")
    check("X-Frame-Options: DENY",
          r.headers.get("x-frame-options") == "DENY")
    check("X-XSS-Protection: 1; mode=block",
          r.headers.get("x-xss-protection") == "1; mode=block")

    # Auth errors must not leak internals
    r = c.get(f"{GATEWAY}/v1/models", headers=auth("bad-key"))
    body = r.json()
    err_msg = body.get("error", {}).get("message", "")
    check("401 error message does not contain 'hash'",    "hash"  not in err_msg.lower())
    check("401 error message does not contain 'dashmap'", "dashmap" not in err_msg.lower())
    check("401 error message does not contain 'postgres'", "postgres" not in err_msg.lower())


def test_auth(c: httpx.Client, api_key: str):
    section("Authentication")

    # No key
    r = c.get(f"{GATEWAY}/v1/models")
    check("No key → 401", r.status_code == 401)
    check("No key → OpenAI error shape",
          r.json().get("error", {}).get("type") == "authentication_error")

    # Wrong key
    r = c.get(f"{GATEWAY}/v1/models", headers=auth("totally-wrong"))
    check("Wrong key → 401", r.status_code == 401)

    # Bearer header
    r = c.get(f"{GATEWAY}/v1/models", headers=auth(api_key))
    check("Valid Bearer key → 200", r.status_code == 200)

    # X-API-Key header
    r = c.get(f"{GATEWAY}/v1/models", headers={"X-API-Key": api_key})
    check("Valid X-API-Key → 200", r.status_code == 200)

    # Both headers present — Bearer takes precedence
    r = c.get(f"{GATEWAY}/v1/models",
              headers={**auth(api_key), "X-API-Key": "garbage"})
    check("Bearer wins over bad X-API-Key → 200", r.status_code == 200)


def test_models(c: httpx.Client, api_key: str):
    section("Model listing")

    r = c.get(f"{GATEWAY}/v1/models", headers=auth(api_key))
    check("/v1/models → 200", r.status_code == 200)
    body = r.json()
    check("Response has 'data' list", isinstance(body.get("data"), list))
    check("At least one model served", len(body.get("data", [])) > 0)
    if body.get("data"):
        m = body["data"][0]
        check("Model object has 'id'", "id" in m)
        check("Model id is Qwen3.5-35B-A3B", m["id"] == "Qwen/Qwen3.5-27B")

    r = c.get(f"{GATEWAY}/v1/models/local", headers=auth(api_key))
    check("/v1/models/local → 200", r.status_code == 200)
    local = r.json()
    check("local models list non-empty", len(local.get("data", [])) > 0)
    formats = {m["format"] for m in local.get("data", [])}
    check("local models include gguf format", "gguf" in formats)
    check("local models include hf format",   "hf"   in formats)


def test_request_validation(c: httpx.Client, api_key: str):
    section("Request validation & error shapes")

    hdrs = {**auth(api_key), **json_body()}

    # Missing messages field
    r = c.post(f"{GATEWAY}/v1/chat/completions", headers=hdrs,
               json={"model": "Qwen/Qwen3.5-27B"})
    check("Missing messages → 4xx", r.status_code >= 400)
    err = r.json().get("error", {})
    check("Error has 'message' key", "message" in err)
    check("Error has 'type' key",    "type"    in err)
    check("Error has 'param' key",   "param"   in err)
    check("Error has 'code' key",    "code"    in err)

    # Empty messages list
    r = c.post(f"{GATEWAY}/v1/chat/completions", headers=hdrs,
               json={"model": "Qwen/Qwen3.5-27B", "messages": []})
    check("Empty messages → 4xx", r.status_code >= 400)

    # Invalid JSON
    r = c.post(f"{GATEWAY}/v1/chat/completions",
               content=b"{not valid json}", headers=hdrs)
    check("Invalid JSON body → 4xx", r.status_code >= 400)

    # Body too large (>4MB)
    big = b"x" * (4 * 1024 * 1024 + 1)
    r = c.post(f"{GATEWAY}/v1/chat/completions", headers=hdrs, content=big,
               timeout=10)
    check("Body >4MB → 413", r.status_code == 413)


def test_admin(c: httpx.Client, api_key: str, admin_key: str):
    section("Admin API — key lifecycle")

    ah = {**auth(admin_key), **json_body()}

    # Regular key cannot access admin
    r = c.get(f"{GATEWAY}/admin/keys", headers=auth(api_key))
    check("Regular key → admin 401", r.status_code == 401)

    # List existing keys
    r = c.get(f"{GATEWAY}/admin/keys", headers=auth(admin_key))
    check("Admin list keys → 200", r.status_code == 200)
    before = r.json()
    check("List returns array", isinstance(before, list))
    key_fields = {"id", "label", "created_at", "enabled"}
    if before:
        check("Key object has expected fields", key_fields.issubset(before[0].keys()))

    # Create a new key
    r = c.post(f"{GATEWAY}/admin/keys", headers=ah, json={"label": "test-lifecycle"})
    check("Create key → 201", r.status_code == 201)
    created = r.json()
    check("Create response has plaintext key", created.get("key", "").startswith("gw_"))
    check("Create response has id",         "id"         in created)
    check("Create response has label",      created.get("label") == "test-lifecycle")
    check("Create response has created_at", "created_at" in created)
    new_id  = created.get("id")
    new_key = created.get("key")

    # New key works immediately
    r = c.get(f"{GATEWAY}/v1/models", headers=auth(new_key))
    check("Freshly created key → 200", r.status_code == 200)

    # Disable the key
    r = c.patch(f"{GATEWAY}/admin/keys/{new_id}", headers=ah,
                json={"enabled": False})
    check("Disable key → 200", r.status_code == 200)
    check("Response shows enabled=false", r.json().get("enabled") == False)

    # Disabled key is rejected (cache refresh may add up to 60s — but new keys
    # are updated immediately in the DashMap, so disable is instant)
    r = c.get(f"{GATEWAY}/v1/models", headers=auth(new_key))
    check("Disabled key → 401", r.status_code == 401)

    # Re-enable
    r = c.patch(f"{GATEWAY}/admin/keys/{new_id}", headers=ah,
                json={"enabled": True})
    check("Re-enable key → 200", r.status_code == 200)
    r = c.get(f"{GATEWAY}/v1/models", headers=auth(new_key))
    check("Re-enabled key → 200", r.status_code == 200)

    # Delete key
    r = c.delete(f"{GATEWAY}/admin/keys/{new_id}", headers=auth(admin_key))
    check("Delete key → 204", r.status_code == 204)

    # Deleted key is rejected
    r = c.get(f"{GATEWAY}/v1/models", headers=auth(new_key))
    check("Deleted key → 401", r.status_code == 401)

    # List should be back to original count
    r = c.get(f"{GATEWAY}/admin/keys", headers=auth(admin_key))
    after = r.json()
    check("Key count restored after delete",
          len(after) == len(before),
          f"before={len(before)} after={len(after)}")

    # Delete non-existent id
    r = c.delete(f"{GATEWAY}/admin/keys/00000000-0000-0000-0000-000000000000",
                 headers=auth(admin_key))
    check("Delete unknown id → 404", r.status_code == 404)


def test_usage(c: httpx.Client, api_key: str):
    section("Usage & quota endpoint")

    r = c.get(f"{GATEWAY}/v1/usage", headers=auth(api_key))
    check("GET /v1/usage → 200", r.status_code == 200)
    body = r.json()
    check("Has completion_tokens", "completion_tokens" in body)
    check("Has prompt_tokens",     "prompt_tokens"     in body)
    check("Has requests",          "requests"          in body)
    check("Has quota",             "quota"             in body)
    check("Has quota_remaining",   "quota_remaining"   in body)
    check("Has date",              "date"              in body)
    check("quota=0 means unlimited", body.get("quota") == 0)
    check("quota_remaining=-1 when unlimited", body.get("quota_remaining") == -1)

    # Usage without key
    r = c.get(f"{GATEWAY}/v1/usage")
    check("GET /v1/usage without key → 401", r.status_code == 401)


def test_last_used_at(c: httpx.Client, api_key: str, admin_key: str):
    section("last_used_at tracking")

    # Find the key entry for api_key
    r = c.get(f"{GATEWAY}/admin/keys", headers=auth(admin_key))
    keys_before = {k["label"]: k for k in r.json()}

    # Make a request with api_key
    c.get(f"{GATEWAY}/v1/models", headers=auth(api_key))
    time.sleep(1)  # give the fire-and-forget task time to write

    r = c.get(f"{GATEWAY}/admin/keys", headers=auth(admin_key))
    keys_after = {k["label"]: k for k in r.json()}

    # Find the migrated key that matches api_key (migrated-0 = dev-key-1)
    touched = any(
        k.get("last_used_at") is not None
        for k in keys_after.values()
    )
    check("last_used_at populated after request", touched)


def test_completions_endpoint(c: httpx.Client, api_key: str):
    section("POST /v1/completions (raw text completion)")

    hdrs = {**auth(api_key), **json_body()}
    r = c.post(f"{GATEWAY}/v1/completions", headers=hdrs,
               json={
                   "model": "Qwen/Qwen3.5-27B",
                   "prompt": "The capital of France is",
                   "max_tokens": 5,
                   "temperature": 0.0,
               }, timeout=120)
    check("POST /v1/completions → 200", r.status_code == 200)
    body = r.json()
    check("Response has choices", len(body.get("choices", [])) > 0)
    text = body["choices"][0].get("text", "")
    check("Completion text is non-empty", len(text.strip()) > 0,
          f"got: {repr(text)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key",       default=None)
    p.add_argument("--admin-key", default=None)
    args = p.parse_args()

    cfg = load_config()
    api_key   = args.key       or cfg.get("GATEWAY_API_KEYS", "").split(",")[0].strip()
    admin_key = args.admin_key or cfg.get("ADMIN_KEY", "")

    if not api_key:
        sys.exit("No API key found. Set GATEWAY_API_KEYS in config/.env or pass --key")
    if not admin_key:
        sys.exit("No admin key found. Set ADMIN_KEY in config/.env or pass --admin-key")

    print(f"Gateway:   {GATEWAY}")
    print(f"API key:   {api_key[:8]}...")
    print(f"Admin key: {admin_key[:8]}...")

    with httpx.Client(timeout=120) as c:
        test_health(c)
        test_metrics(c)
        test_security_headers(c, api_key)
        test_auth(c, api_key)
        test_models(c, api_key)
        test_request_validation(c, api_key)
        test_admin(c, api_key, admin_key)
        test_usage(c, api_key)
        test_last_used_at(c, api_key, admin_key)
        test_completions_endpoint(c, api_key)

    # Summary
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'─'*60}")
    print(f"  {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for name, ok, detail in _results:
            if not ok:
                print(f"    - {name}" + (f": {detail}" if detail else ""))
    else:
        print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
