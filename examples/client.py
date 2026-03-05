#!/usr/bin/env python3
"""
Simple client for the LLM inference stack.
Talks to either the llama-cpp-python server (port 8001) or the Rust gateway (port 8080).

API key resolution order:
  1. --key CLI flag
  2. GATEWAY_API_KEYS env var (first key in comma-separated list)
  3. config/.env in the repo root (parsed directly)
  Fails with an error if none of the above provide a key.

Usage:
    uv run python3 examples/client.py "your prompt"
    uv run python3 examples/client.py --stream "your prompt"
    uv run python3 examples/client.py --url http://localhost:8080 "your prompt"
    uv run python3 examples/client.py --raw "your prompt"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

# Repo root is one level up from examples/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _REPO_ROOT / "config" / ".env"


def load_env_file(path: Path) -> dict:
    """Parse a simple KEY=value .env file (no shell expansion)."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def resolve_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key

    # env var
    if val := os.environ.get("GATEWAY_API_KEYS"):
        return val.split(",")[0].strip()

    # .env file
    env = load_env_file(_ENV_FILE)
    if val := env.get("GATEWAY_API_KEYS"):
        return val.split(",")[0].strip()

    sys.exit(
        "error: no API key found.\n"
        "  Set GATEWAY_API_KEYS in config/.env, export it in your shell,\n"
        "  or pass --key <key>.\n"
        "  (For llama-cpp-python on port 8001 any key works — set GATEWAY_API_KEYS=any)"
    )


def parse_args():
    p = argparse.ArgumentParser(description="LLM inference client")
    p.add_argument("prompt", nargs="*", default=["What is tensor parallelism?"],
                   help="Prompt text (default: 'What is tensor parallelism?')")
    p.add_argument("--url", default="http://localhost:8001",
                   help="Base URL of the server (default: http://localhost:8001)")
    p.add_argument("--model", default=None,
                   help="Model ID. Auto-detected from /v1/models if not set.")
    p.add_argument("--key", default=None,
                   help="API key. Falls back to GATEWAY_API_KEYS from config/.env.")
    p.add_argument("--stream", action="store_true",
                   help="Use streaming mode")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--raw", action="store_true",
                   help="Print the raw JSON response instead of just the content")
    p.add_argument("--no-think", action="store_true", default=True,
                   help="Suppress Qwen3 thinking tokens (default: on)")
    return p.parse_args()


def get_model(base_url: str, headers: dict) -> str:
    r = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=10)
    r.raise_for_status()
    models = r.json()["data"]
    if not models:
        raise RuntimeError("No models available on server")
    return models[0]["id"]


def chat_sync(base_url: str, headers: dict, model: str, messages: list,
              max_tokens: int, temperature: float, raw: bool):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = httpx.post(f"{base_url}/v1/chat/completions", headers=headers,
                   json=payload, timeout=120)
    r.raise_for_status()
    body = r.json()

    if raw:
        print(json.dumps(body, indent=2))
    else:
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        print(content)
        print(f"\n[tokens: {usage.get('prompt_tokens','?')}p + "
              f"{usage.get('completion_tokens','?')}c = "
              f"{usage.get('total_tokens','?')} total]")


def chat_stream(base_url: str, headers: dict, model: str, messages: list,
                max_tokens: int, temperature: float, raw: bool):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    with httpx.stream("POST", f"{base_url}/v1/chat/completions",
                      headers=headers, json=payload, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if raw:
                print(json.dumps(chunk))
            else:
                delta = chunk["choices"][0]["delta"].get("content") or ""
                print(delta, end="", flush=True)
    if not raw:
        print()


def main():
    args = parse_args()
    api_key = resolve_api_key(args.key)
    prompt = " ".join(args.prompt)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    model = args.model or get_model(args.url, headers)
    print(f"[server: {args.url}]  [model: {model}]\n", file=sys.stderr)

    messages = []
    if args.no_think:
        messages.append({"role": "system", "content": "/no_think"})
    messages.append({"role": "user", "content": prompt})

    if args.stream:
        chat_stream(args.url, headers, model, messages,
                    args.max_tokens, args.temperature, args.raw)
    else:
        chat_sync(args.url, headers, model, messages,
                  args.max_tokens, args.temperature, args.raw)


if __name__ == "__main__":
    main()
