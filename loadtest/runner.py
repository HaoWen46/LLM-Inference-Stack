#!/usr/bin/env python3
"""
Load tester for the LLM gateway.

Usage:
    python3 loadtest/runner.py --url http://localhost:8080 --key dev-key-1 \
        --concurrency 16 --duration 120 --model llama-7b

Output:
    Live progress via Rich, final report with P50/P95/P99 latency,
    TTFT, throughput, and error breakdown.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# ---------------------------------------------------------------------------
# Prompt workload distribution
# ---------------------------------------------------------------------------
PROMPTS = [
    # (label, prompt_text, max_tokens)
    ("short_qa",
     "What is the boiling point of water in Celsius?",
     32),
    ("medium_explain",
     "Explain the difference between TCP and UDP, and give two use cases for each.",
     256),
    ("code_gen",
     "Write a Python function that implements binary search on a sorted list. "
     "Include type hints and a docstring.",
     300),
    ("reasoning",
     "A train leaves city A at 9am travelling at 80km/h. Another leaves city B "
     "at 10am travelling at 100km/h toward city A. The cities are 500km apart. "
     "At what time do they meet? Show your work step by step.",
     400),
    ("long_form",
     "Write a detailed technical design document for a distributed rate limiter "
     "using Redis. Cover data structures, expiry strategy, race conditions, "
     "and failover behavior.",
     800),
]

WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    request_id: str
    prompt_label: str
    start: float
    ttft: float = 0.0
    end: float = 0.0
    tokens_out: int = 0
    prompt_tokens: int = 0
    error: str = ""
    status_code: int = 200

    @property
    def latency(self) -> float:
        return self.end - self.start

    @property
    def decode_tps(self) -> float:
        decode_s = self.end - (self.start + self.ttft)
        return self.tokens_out / decode_s if decode_s > 0 and self.tokens_out > 0 else 0.0

    @property
    def ok(self) -> bool:
        return not self.error and self.status_code == 200


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------
async def single_request(
    client: httpx.AsyncClient,
    url_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    label: str,
) -> RequestResult:
    rid = f"lt-{time.monotonic_ns()}"
    result = RequestResult(request_id=rid, prompt_label=label, start=time.monotonic())

    try:
        first_token_seen = False
        token_count = 0

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": 0.7,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0),
        ) as resp:
            result.status_code = resp.status_code

            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}"
                result.end = time.monotonic()
                return result

            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue

                if not first_token_seen:
                    result.ttft = time.monotonic() - result.start
                    first_token_seen = True

                try:
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        token_count += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        result.end = time.monotonic()
        result.tokens_out = token_count

    except httpx.TimeoutException:
        result.end = time.monotonic()
        result.error = "timeout"
        result.status_code = 504
    except httpx.RequestError as exc:
        result.end = time.monotonic()
        result.error = f"connection_error: {exc}"
        result.status_code = 503
    except Exception as exc:
        result.end = time.monotonic()
        result.error = f"unexpected: {exc}"

    return result


# ---------------------------------------------------------------------------
# Load driver
# ---------------------------------------------------------------------------
async def run_load_test(
    url: str,
    api_key: str,
    model: str,
    concurrency: int,
    duration: float,
    ramp_up: float = 10.0,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(base_url=url, http2=True) as client:
        # Warmup
        console.print(f"[bold yellow]Warming up (3 requests)...[/]")
        warmup_tasks = [
            single_request(client, url, api_key, model, PROMPTS[0][1], PROMPTS[0][2], "warmup")
            for _ in range(3)
        ]
        warmup_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
        for r in warmup_results:
            if isinstance(r, RequestResult) and r.ok:
                console.print(f"  warmup ok — TTFT {r.ttft*1000:.0f}ms, latency {r.latency:.2f}s")
            else:
                console.print(f"  [red]warmup failed: {r}[/]")

        console.print(f"\n[bold green]Starting load test:[/] concurrency={concurrency}, duration={duration}s")

        end_time = time.monotonic() + duration
        semaphore = asyncio.Semaphore(concurrency)
        done = asyncio.Event()

        async def worker() -> None:
            while time.monotonic() < end_time:
                async with semaphore:
                    label, prompt, max_tok = random.choices(PROMPTS, weights=WEIGHTS, k=1)[0]
                    r = await single_request(client, url, api_key, model, prompt, max_tok, label)
                    async with lock:
                        results.append(r)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency * 3)]

        # Live stats display
        with Live(console=console, refresh_per_second=2) as live:
            while time.monotonic() < end_time:
                await asyncio.sleep(0.5)
                live.update(_make_live_table(results, concurrency, end_time))

        await asyncio.gather(*tasks, return_exceptions=True)

    return results


def _make_live_table(results: list[RequestResult], concurrency: int, end_time: float) -> Table:
    ok = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]
    elapsed = max(time.monotonic() - (end_time - (end_time - time.monotonic())), 0.01)

    table = Table(title="[bold]Live Load Test Stats[/]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Completed requests", str(len(results)))
    table.add_row("Successful", str(len(ok)))
    table.add_row("Errors", f"[red]{len(err)}[/]" if err else "0")
    table.add_row("Concurrency", str(concurrency))

    if ok:
        lats = np.array([r.latency for r in ok])
        ttfts = np.array([r.ttft for r in ok if r.ttft > 0])
        toks = sum(r.tokens_out for r in ok)
        wall = ok[-1].end - ok[0].start if len(ok) > 1 else 1.0

        table.add_row("Req/s", f"{len(ok)/wall:.2f}")
        table.add_row("Agg tok/s", f"{toks/wall:.1f}")
        table.add_row("Latency P50", f"{np.percentile(lats, 50):.2f}s")
        table.add_row("Latency P95", f"{np.percentile(lats, 95):.2f}s")
        if len(ttfts) > 0:
            table.add_row("TTFT P50", f"{np.percentile(ttfts, 50)*1000:.0f}ms")
            table.add_row("TTFT P95", f"{np.percentile(ttfts, 95)*1000:.0f}ms")

    table.add_row("Time remaining", f"{max(0, end_time - time.monotonic()):.0f}s")
    return table


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
def print_report(results: list[RequestResult]) -> None:
    ok    = [r for r in results if r.ok]
    errs  = [r for r in results if not r.ok]

    if not results:
        console.print("[red]No results collected.[/]")
        return

    console.rule("[bold]Load Test Report[/]")

    # Summary
    wall = ok[-1].end - ok[0].start if len(ok) > 1 else 1.0
    total_toks = sum(r.tokens_out for r in ok)

    console.print(f"\n[bold]Summary[/]")
    console.print(f"  Total requests   : {len(results)}")
    console.print(f"  Successful       : {len(ok)}  ({100*len(ok)/len(results):.1f}%)")
    console.print(f"  Failed           : {len(errs)}")
    console.print(f"  Wall time        : {wall:.1f}s")
    console.print(f"  Requests/sec     : {len(ok)/wall:.2f}")
    console.print(f"  Agg tokens/sec   : {total_toks/wall:.1f}")

    if ok:
        lats  = np.array([r.latency for r in ok])
        ttfts = np.array([r.ttft for r in ok if r.ttft > 0])
        tps   = np.array([r.decode_tps for r in ok if r.decode_tps > 0])

        console.print(f"\n[bold]End-to-End Latency[/]")
        for p in [50, 90, 95, 99]:
            console.print(f"  P{p:2d}  : {np.percentile(lats, p):.3f}s")

        if len(ttfts) > 0:
            console.print(f"\n[bold]Time To First Token[/]")
            for p in [50, 90, 95, 99]:
                console.print(f"  P{p:2d}  : {np.percentile(ttfts, p)*1000:.0f}ms")

        if len(tps) > 0:
            console.print(f"\n[bold]Decode Throughput (per-request tok/s)[/]")
            for p in [10, 50, 90]:
                console.print(f"  P{p:2d}  : {np.percentile(tps, p):.1f} tok/s")

        # Per-prompt-type breakdown
        console.print(f"\n[bold]By Prompt Type[/]")
        labels = sorted({r.prompt_label for r in ok})
        t = Table()
        t.add_column("Label"); t.add_column("N"); t.add_column("P50 lat"); t.add_column("P95 lat"); t.add_column("P50 TTFT")
        for label in labels:
            subset = [r for r in ok if r.prompt_label == label]
            if not subset:
                continue
            sl = np.array([r.latency for r in subset])
            st = np.array([r.ttft for r in subset if r.ttft > 0])
            t.add_row(
                label, str(len(subset)),
                f"{np.percentile(sl, 50):.2f}s",
                f"{np.percentile(sl, 95):.2f}s",
                f"{np.percentile(st, 50)*1000:.0f}ms" if len(st) else "n/a",
            )
        console.print(t)

    if errs:
        console.print(f"\n[bold red]Errors[/]")
        from collections import Counter
        for err, count in Counter(r.error for r in errs).most_common(10):
            console.print(f"  {count:4d}x  {err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Gateway Load Tester")
    parser.add_argument("--url",         default="http://localhost:8080")
    parser.add_argument("--key",         default=os.environ.get("GATEWAY_API_KEYS", "dev-key-1").split(",")[0])
    parser.add_argument("--model",       default=os.environ.get("SERVED_MODEL_NAME", "llama-7b"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--duration",    type=float, default=60.0, help="Test duration in seconds")
    parser.add_argument("--ramp-up",     type=float, default=10.0)
    args = parser.parse_args()

    results = asyncio.run(run_load_test(
        url=args.url,
        api_key=args.key,
        model=args.model,
        concurrency=args.concurrency,
        duration=args.duration,
        ramp_up=args.ramp_up,
    ))

    print_report(results)


if __name__ == "__main__":
    main()
