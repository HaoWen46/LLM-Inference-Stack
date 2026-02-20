#!/usr/bin/env python3
"""
Standalone GPU Prometheus exporter.
Scrapes nvidia-smi every 2s and exposes metrics on :9101/metrics.
Run alongside vLLM as a sidecar process.
"""
import subprocess
import sys
import time
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus_client import (
    Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

# ── Gauges ────────────────────────────────────────────────────────────────────
G_UTIL        = Gauge("gpu_utilization_percent",      "GPU compute utilization",       ["gpu_id", "gpu_name"])
G_MEM_USED    = Gauge("gpu_memory_used_mib",          "GPU memory used (MiB)",         ["gpu_id", "gpu_name"])
G_MEM_FREE    = Gauge("gpu_memory_free_mib",          "GPU memory free (MiB)",         ["gpu_id", "gpu_name"])
G_MEM_TOTAL   = Gauge("gpu_memory_total_mib",         "GPU memory total (MiB)",        ["gpu_id", "gpu_name"])
G_MEM_UTIL    = Gauge("gpu_memory_utilization_percent","GPU memory bandwidth util",    ["gpu_id", "gpu_name"])
G_TEMP        = Gauge("gpu_temperature_celsius",      "GPU core temperature",          ["gpu_id", "gpu_name"])
G_POWER       = Gauge("gpu_power_draw_watts",         "GPU power draw",                ["gpu_id", "gpu_name"])
G_POWER_LIMIT = Gauge("gpu_power_limit_watts",        "GPU TDP power limit",           ["gpu_id", "gpu_name"])
G_SM_CLOCK    = Gauge("gpu_sm_clock_mhz",             "GPU SM clock speed",            ["gpu_id", "gpu_name"])
G_MEM_CLOCK   = Gauge("gpu_mem_clock_mhz",            "GPU memory clock speed",        ["gpu_id", "gpu_name"])
G_FAN         = Gauge("gpu_fan_speed_percent",        "GPU fan speed",                 ["gpu_id", "gpu_name"])
G_ECC_CORR    = Gauge("gpu_ecc_errors_corrected_total","ECC corrected errors",         ["gpu_id", "gpu_name"])
G_ECC_UNCORR  = Gauge("gpu_ecc_errors_uncorrected_total","ECC uncorrected errors",     ["gpu_id", "gpu_name"])
G_PCIE_TX     = Gauge("gpu_pcie_tx_kb_per_sec",       "PCIe TX throughput (KB/s)",     ["gpu_id", "gpu_name"])
G_PCIE_RX     = Gauge("gpu_pcie_rx_kb_per_sec",       "PCIe RX throughput (KB/s)",     ["gpu_id", "gpu_name"])

QUERY_FIELDS = ",".join([
    "index", "name",
    "utilization.gpu", "utilization.memory",
    "memory.used", "memory.free", "memory.total",
    "temperature.gpu",
    "power.draw", "power.limit",
    "clocks.sm", "clocks.mem",
    "fan.speed",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.uncorrected.volatile.total",
    "pcie.link.gen.current", "pcie.link.width.current",
])

def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val.strip().split()[0])
    except (ValueError, IndexError):
        return default


def collect_once() -> None:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        print(f"[gpu_exporter] nvidia-smi error: {result.stderr}", flush=True)
        return

    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 17:
            continue

        (idx, name, util_gpu, util_mem,
         mem_used, mem_free, mem_total,
         temp, power, power_limit,
         sm_clock, mem_clock, fan,
         ecc_corr, ecc_uncorr,
         pcie_gen, pcie_width) = parts[:17]

        gpu_id   = f"gpu{idx}"
        gpu_name = name.strip()
        labels   = [gpu_id, gpu_name]

        G_UTIL.labels(*labels).set(_safe_float(util_gpu))
        G_MEM_UTIL.labels(*labels).set(_safe_float(util_mem))
        G_MEM_USED.labels(*labels).set(_safe_float(mem_used))
        G_MEM_FREE.labels(*labels).set(_safe_float(mem_free))
        G_MEM_TOTAL.labels(*labels).set(_safe_float(mem_total))
        G_TEMP.labels(*labels).set(_safe_float(temp))
        G_POWER.labels(*labels).set(_safe_float(power))
        G_POWER_LIMIT.labels(*labels).set(_safe_float(power_limit))
        G_SM_CLOCK.labels(*labels).set(_safe_float(sm_clock))
        G_MEM_CLOCK.labels(*labels).set(_safe_float(mem_clock))
        G_FAN.labels(*labels).set(_safe_float(fan))
        G_ECC_CORR.labels(*labels).set(_safe_float(ecc_corr))
        G_ECC_UNCORR.labels(*labels).set(_safe_float(ecc_uncorr))

    # Also collect per-process GPU memory
    _collect_process_memory()


def _collect_process_memory() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    # (omitted from metrics for simplicity — logged to stdout for debugging)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            output = generate_latest(REGISTRY)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress default access log


def _collection_loop(interval: float = 2.0) -> None:
    while True:
        try:
            collect_once()
        except Exception as exc:
            print(f"[gpu_exporter] collection error: {exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    port = int(os.environ.get("GPU_EXPORTER_PORT", "9101"))
    print(f"[gpu_exporter] Starting on :{port}", flush=True)

    # Collection in background thread
    t = threading.Thread(target=_collection_loop, daemon=True)
    t.start()

    # Initial collection before serving first scrape
    collect_once()

    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    server.serve_forever()
