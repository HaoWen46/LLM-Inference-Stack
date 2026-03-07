#!/usr/bin/env python3
"""
Quick smoke test for the gateway + vLLM stack.
Sends a few requests with system prompts and prints results.

Usage:
    .venv/bin/python3 examples/test_gateway.py
    .venv/bin/python3 examples/test_gateway.py --key gw_...
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:8080"
_ENV_FILE = Path(__file__).resolve().parent.parent / "config" / ".env"


def load_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("GATEWAY_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip().strip('"\'')
    sys.exit("No API key found. Pass --key or set GATEWAY_API_KEYS in config/.env")


def headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def chat(client: httpx.Client, key: str, system: str, user: str,
         max_tokens: int = 512, think: bool = False,
         model: str = "Qwen/Qwen3.5-27B") -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
    }
    if not think:
        # vLLM Qwen3 reasoning parser honours this extra_body flag to skip <think>
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    r = client.post(
        f"{GATEWAY}/v1/chat/completions",
        headers=headers(key),
        json=payload,
        timeout=300,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    msg = body["choices"][0]["message"]
    # content is null when reasoning parser strips <think> and only reasoning_content remains
    content = (msg.get("content") or msg.get("reasoning") or "").strip()
    usage = body.get("usage", {})
    usage["elapsed_s"] = round(elapsed, 2)
    usage["tok_per_s"] = round(usage.get("completion_tokens", 0) / elapsed, 1)
    return content, usage


def chat_stream(client: httpx.Client, key: str, system: str, user: str,
                max_tokens: int = 300) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    chunks = []
    t0 = time.perf_counter()
    with client.stream(
        "POST",
        f"{GATEWAY}/v1/chat/completions",
        headers=headers(key),
        json={
            "model": "Qwen/Qwen3.5-27B",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0]["delta"].get("content") or ""
            chunks.append(delta)
            print(delta, end="", flush=True)
    print()
    elapsed = time.perf_counter() - t0
    return "".join(chunks), elapsed


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run(key: str):
    with httpx.Client() as client:

        # ── Health ────────────────────────────────────────────────
        section("1. Health checks")
        for name, url in [("gateway /health", f"{GATEWAY}/health"),
                          ("gateway /ready",  f"{GATEWAY}/ready"),
                          ("vllm models",     f"{GATEWAY}/v1/models")]:
            r = client.get(url, headers=headers(key), timeout=10)
            status = "OK" if r.is_success else f"FAIL ({r.status_code})"
            print(f"  {name}: {status}")

        # ── Basic Q&A with system prompt ──────────────────────────
        section("2. Basic Q&A — system prompt sets persona")
        content, usage = chat(
            client, key,
            system="You are a terse Unix wizard. Answer in at most two sentences. No fluff.",
            user="What does 'chmod 755' do?",
        )
        print(content)
        print(f"\n  [{usage['prompt_tokens']}p + {usage['completion_tokens']}c "
              f"= {usage['total_tokens']} tokens | {usage['elapsed_s']}s | "
              f"{usage['tok_per_s']} tok/s]")

        # ── Instruction following ─────────────────────────────────
        section("3. Instruction following — structured output")
        content, usage = chat(
            client, key,
            system="You are a helpful assistant. Always respond with valid JSON only. No prose, no markdown fences.",
            user='Give me a JSON object with keys "language", "year_created", and "creator" for Python.',
            max_tokens=512,
        )
        print(content)
        try:
            parsed = json.loads(content)
            print(f"\n  JSON parsed OK: {parsed}")
        except json.JSONDecodeError as e:
            print(f"\n  WARNING: response is not valid JSON: {e}")
        print(f"  [{usage['prompt_tokens']}p + {usage['completion_tokens']}c | {usage['elapsed_s']}s]")

        # ── Streaming ─────────────────────────────────────────────
        section("4. Streaming response")
        print("  (tokens arriving live below)\n  ", end="")
        content, elapsed = chat_stream(
            client, key,
            system="You are a helpful assistant. Be concise.",
            user="Name three benefits of tensor parallelism for large language models.",
            max_tokens=200,
        )
        print(f"  [elapsed: {elapsed:.2f}s]")

        # ── Reasoning mode (think=True) ───────────────────────────
        section("5. Reasoning mode — model thinks before answering")
        print("  (thinking tokens are stripped by --reasoning-parser qwen3;")
        print("   only the final answer appears in 'content')\n")
        content, usage = chat(
            client, key,
            system="You are a careful reasoning assistant.",
            user="A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
            max_tokens=1024,
            think=True,
        )
        print(content)
        print(f"\n  [{usage['prompt_tokens']}p + {usage['completion_tokens']}c | {usage['elapsed_s']}s]")

        # ── LoRA adapter ──────────────────────────────────────────
        section("6. LoRA adapter — qwen3-27b-opus")
        print("  Sending request with model='qwen3-27b-opus' to activate the LoRA adapter.\n")
        content, usage = chat(
            client, key,
            system="You are a helpful assistant.",
            user="In one sentence, what is a LoRA adapter in the context of LLMs?",
            max_tokens=128,
            think=False,
            model="qwen3-27b-opus",
        )
        print(content)
        print(f"\n  [{usage['prompt_tokens']}p + {usage['completion_tokens']}c | {usage['elapsed_s']}s | {usage['tok_per_s']} tok/s]")

        # ── Tokenize / Detokenize ─────────────────────────────────────
        section("7. Tokenize + Detokenize")
        tok_payload = {
            "model": "Qwen/Qwen3.5-27B",
            "prompt": "Hello, world! This is a tokenization test.",
        }
        r = client.post(f"{GATEWAY}/v1/tokenize", headers=headers(key), json=tok_payload, timeout=15)
        if r.is_success:
            tok_body = r.json()
            tokens = tok_body.get("tokens", [])
            print(f"  tokenize: {len(tokens)} tokens → {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            det_payload = {"model": "Qwen/Qwen3.5-27B", "tokens": tokens}
            r2 = client.post(f"{GATEWAY}/v1/detokenize", headers=headers(key), json=det_payload, timeout=15)
            if r2.is_success:
                recovered = r2.json().get("prompt", "")
                print(f"  detokenize: '{recovered}'")
            else:
                print(f"  detokenize FAIL ({r2.status_code}): {r2.text[:200]}")
        else:
            print(f"  tokenize FAIL ({r.status_code}): {r.text[:200]}")

        # ── Embeddings ────────────────────────────────────────────────
        section("8. Embeddings")
        emb_payload = {"model": "Qwen/Qwen3.5-27B", "input": "The sky is blue."}
        r = client.post(f"{GATEWAY}/v1/embeddings", headers=headers(key), json=emb_payload, timeout=30)
        if r.is_success:
            emb = r.json()["data"][0]["embedding"]
            print(f"  embedding dim: {len(emb)}  first 5: {emb[:5]}")
        else:
            print(f"  NOTE: embeddings not supported with this model (status {r.status_code})")
            print(f"  Response: {r.text[:300]}")

        # ── Batch API ─────────────────────────────────────────────────
        section("9. Batch API")
        batch_payload = {
            "requests": [
                {
                    "custom_id": "req-1",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "Qwen/Qwen3.5-27B",
                        "messages": [
                            {"role": "system", "content": "One sentence only."},
                            {"role": "user", "content": "What is 2 + 2?"},
                        ],
                        "max_tokens": 32,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
                {
                    "custom_id": "req-2",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "Qwen/Qwen3.5-27B",
                        "messages": [
                            {"role": "system", "content": "One sentence only."},
                            {"role": "user", "content": "Name the capital of France."},
                        ],
                        "max_tokens": 32,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
            ],
        }
        r = client.post(f"{GATEWAY}/v1/batches", headers=headers(key), json=batch_payload, timeout=15)
        if not r.is_success:
            print(f"  create batch FAIL ({r.status_code}): {r.text[:300]}")
        else:
            batch = r.json()
            batch_id = batch["id"]
            print(f"  batch created: {batch_id}  status={batch['status']}  total={batch['total_requests']}")
            # Poll until complete (up to 120s)
            for attempt in range(24):
                time.sleep(5)
                r2 = client.get(f"{GATEWAY}/v1/batches/{batch_id}", headers=headers(key), timeout=10)
                b2 = r2.json()
                status = b2["status"]
                print(f"  poll [{attempt+1}]: status={status}  done={b2['completed_count']}/{b2['total_requests']}")
                if status in ("completed", "failed", "cancelled"):
                    break
            # Fetch results
            r3 = client.get(f"{GATEWAY}/v1/batches/{batch_id}/results", headers=headers(key), timeout=10)
            if r3.is_success:
                for item in r3.json().get("data", []):
                    cid = item["custom_id"]
                    answer = (
                        item["body"]["choices"][0]["message"].get("content", "").strip()
                        if item.get("body")
                        else f"ERROR: {item.get('error')}"
                    )
                    print(f"  {cid}: {answer}")
            else:
                print(f"  results FAIL ({r3.status_code}): {r3.text[:200]}")

        # ── Usage endpoint ────────────────────────────────────────
        section("10. Per-key usage quota")
        r = client.get(f"{GATEWAY}/v1/usage", headers=headers(key), timeout=10)
        print(json.dumps(r.json(), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", default=None, help="API key (default: from config/.env)")
    args = p.parse_args()
    key = load_api_key(args.key)
    print(f"Gateway: {GATEWAY}")
    print(f"Key:     {key[:8]}...")
    run(key)


if __name__ == "__main__":
    main()
