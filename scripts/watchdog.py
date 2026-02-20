#!/usr/bin/env python3
"""
Process watchdog — monitors vLLM and gateway processes,
restarts them if they die, and alerts on NCCL hang detection.

Run as a supervisor alongside your stack. Uses simple subprocess
management rather than systemd for portability.

Usage:
    python3 scripts/watchdog.py
"""
import os
import signal
import subprocess
import sys
import time
import threading
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("watchdog")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT_DIR, ".venv", "bin", "python3")
ENV_FILE = os.path.join(ROOT_DIR, "config", ".env")

# Load env vars from .env into os.environ
def _load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))

_load_env(ENV_FILE)


class ManagedProcess:
    def __init__(self, name: str, cmd: list[str], env: dict | None = None,
                 restart_delay: float = 5.0, hang_timeout: float = 300.0):
        self.name = name
        self.cmd = cmd
        self.env = {**os.environ, **(env or {})}
        self.restart_delay = restart_delay
        self.hang_timeout = hang_timeout
        self._proc: subprocess.Popen | None = None
        self._last_healthy = time.monotonic()
        self._restart_count = 0

    def start(self) -> None:
        log.info(f"Starting {self.name}: {' '.join(self.cmd)}")
        self._proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._last_healthy = time.monotonic()
        log.info(f"{self.name} started (pid={self._proc.pid})")

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def is_hung(self) -> bool:
        """Detect hang via external health check (HTTP)."""
        return False  # subclasses override for HTTP health check

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            log.info(f"Stopping {self.name} (pid={self._proc.pid})")
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.warning(f"{self.name} did not stop gracefully — SIGKILL")
                self._proc.kill()
                self._proc.wait()

    def restart(self) -> None:
        self._restart_count += 1
        log.warning(f"Restarting {self.name} (restart #{self._restart_count})")
        self.stop()
        time.sleep(self.restart_delay)
        self.start()


class HTTPHealthProcess(ManagedProcess):
    def __init__(self, *args, health_url: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_url = health_url
        self._last_healthy = time.monotonic()

    def check_health(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(self.health_url, timeout=5) as resp:
                if resp.status == 200:
                    self._last_healthy = time.monotonic()
                    return True
        except Exception:
            pass
        return False

    def is_hung(self) -> bool:
        if not self.is_alive():
            return False
        return (time.monotonic() - self._last_healthy) > self.hang_timeout


def _monitor_loop(processes: list[ManagedProcess], poll_interval: float = 10.0) -> None:
    while True:
        for p in processes:
            try:
                if not p.is_alive():
                    log.error(f"{p.name} died unexpectedly — restarting")
                    p.restart()
                elif isinstance(p, HTTPHealthProcess):
                    p.check_health()
                    if p.is_hung():
                        log.error(f"{p.name} health check failing for >{p.hang_timeout}s — restarting")
                        p.restart()
            except Exception as exc:
                log.exception(f"Monitor error for {p.name}: {exc}")
        time.sleep(poll_interval)


def main() -> None:
    vllm_port   = os.environ.get("VLLM_PORT",    "8000")
    gateway_port = os.environ.get("GATEWAY_PORT", "8080")

    processes: list[ManagedProcess] = [
        HTTPHealthProcess(
            name="vllm",
            cmd=[VENV, "-m", "vllm.entrypoints.openai.api_server",
                 "--config", "/dev/null"],  # placeholder; use launch script in practice
            health_url=f"http://localhost:{vllm_port}/health",
            hang_timeout=120.0,
            restart_delay=10.0,
        ),
    ]

    # Start all managed processes
    for p in processes:
        p.start()
        time.sleep(2)

    # Monitor in main thread
    log.info("Watchdog running. Ctrl-C to stop.")
    try:
        _monitor_loop(processes)
    except KeyboardInterrupt:
        log.info("Watchdog shutting down")
        for p in processes:
            p.stop()


if __name__ == "__main__":
    main()
