"""
Circuit breaker for the vLLM upstream.

State machine:
  CLOSED    — normal operation; failures increment counter
  OPEN      — upstream considered down; all calls fail fast (HTTP 503)
  HALF_OPEN — recovery probe; one request allowed through to test health

Transitions:
  CLOSED  → OPEN      when consecutive failures ≥ failure_threshold
  OPEN    → HALF_OPEN after recovery_timeout seconds
  HALF_OPEN → CLOSED  on first successful probe
  HALF_OPEN → OPEN    on probe failure
"""

import asyncio
import time
from enum import Enum

import structlog

from gateway import metrics

log = structlog.get_logger(__name__)


class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        metrics.CIRCUIT_BREAKER_STATE.set(self._state.value)

    @property
    def state(self) -> CircuitState:
        return self._state

    def _set_state(self, new_state: CircuitState) -> None:
        if new_state != self._state:
            log.info(
                "circuit_breaker_transition",
                old=self._state.name,
                new=new_state.name,
            )
            self._state = new_state
            metrics.CIRCUIT_BREAKER_STATE.set(new_state.value)

    async def _check_state(self) -> None:
        """Evaluate whether to transition from OPEN → HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0)
            if elapsed >= self.recovery_timeout:
                self._set_state(CircuitState.HALF_OPEN)
                self._half_open_calls = 0

    async def before_call(self) -> None:
        """Call before making upstream request. Raises CircuitOpenError if blocked."""
        async with self._lock:
            await self._check_state()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError("Circuit is OPEN — upstream considered unhealthy")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError("Circuit is HALF_OPEN — probe already in flight")
                self._half_open_calls += 1

    async def on_success(self) -> None:
        """Call after a successful upstream response."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                log.info("circuit_breaker_recovered")
            self._set_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._opened_at = None

    async def on_failure(self) -> None:
        """Call after a failed upstream response (timeout, connection error, 5xx)."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — reopen immediately
                self._failure_count = self.failure_threshold
                self._opened_at = time.monotonic()
                self._set_state(CircuitState.OPEN)
                return

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                log.warning(
                    "circuit_breaker_failure",
                    count=self._failure_count,
                    threshold=self.failure_threshold,
                )
                if self._failure_count >= self.failure_threshold:
                    self._opened_at = time.monotonic()
                    self._set_state(CircuitState.OPEN)
