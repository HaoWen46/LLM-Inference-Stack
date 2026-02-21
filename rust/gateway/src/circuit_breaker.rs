/// Lockless circuit breaker using atomics for state transitions.
///
/// States:
///   CLOSED (0)    — normal operation; failures increment counter
///   OPEN (1)      — upstream considered down; all calls fail fast
///   HALF_OPEN (2) — recovery probe; limited calls allowed through
///
/// Transitions:
///   CLOSED → OPEN      when failure_count >= failure_threshold
///   OPEN → HALF_OPEN   after recovery_timeout_secs have elapsed
///   HALF_OPEN → CLOSED on first successful probe
///   HALF_OPEN → OPEN   on probe failure
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::error::GatewayError;

pub const CLOSED: u8 = 0;
pub const OPEN: u8 = 1;
pub const HALF_OPEN: u8 = 2;

pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU64,
    /// Unix epoch milliseconds when the circuit opened; -1 if not open.
    opened_at_ms: AtomicI64,
    half_open_calls: AtomicU64,

    // Config (read-only after construction)
    pub failure_threshold: u64,
    pub recovery_timeout_ms: i64,
    pub half_open_max_calls: u64,
}

impl CircuitBreaker {
    pub fn new(
        failure_threshold: u64,
        recovery_timeout_secs: u64,
        half_open_max_calls: u64,
    ) -> Self {
        Self {
            state: AtomicU8::new(CLOSED),
            failure_count: AtomicU64::new(0),
            opened_at_ms: AtomicI64::new(-1),
            half_open_calls: AtomicU64::new(0),
            failure_threshold,
            recovery_timeout_ms: (recovery_timeout_secs * 1000) as i64,
            half_open_max_calls,
        }
    }

    pub fn state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    /// Check if OPEN long enough → transition to HALF_OPEN.
    fn maybe_transition_to_half_open(&self) {
        if self.state.load(Ordering::Acquire) != OPEN {
            return;
        }
        let opened_at = self.opened_at_ms.load(Ordering::Acquire);
        if opened_at < 0 {
            return;
        }
        if Self::now_ms() - opened_at >= self.recovery_timeout_ms {
            if self
                .state
                .compare_exchange(OPEN, HALF_OPEN, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.half_open_calls.store(0, Ordering::Release);
                info!(state = "HALF_OPEN", "circuit breaker transitioning to half-open");
            }
        }
    }

    /// Call before making an upstream request.
    /// Returns `Err(GatewayError::CircuitOpen)` when the call should be blocked.
    pub fn before_call(&self) -> Result<(), GatewayError> {
        self.maybe_transition_to_half_open();

        match self.state.load(Ordering::Acquire) {
            OPEN => Err(GatewayError::CircuitOpen),
            HALF_OPEN => {
                // Atomically claim a half-open slot.
                let prev = self.half_open_calls.fetch_add(1, Ordering::AcqRel);
                if prev >= self.half_open_max_calls {
                    self.half_open_calls.fetch_sub(1, Ordering::AcqRel);
                    Err(GatewayError::CircuitOpen)
                } else {
                    Ok(())
                }
            }
            _ => Ok(()), // CLOSED
        }
    }

    /// Call after a successful upstream response.
    pub fn on_success(&self) {
        let prev = self.state.load(Ordering::Acquire);
        if prev == HALF_OPEN {
            info!(state = "CLOSED", "circuit breaker recovered");
        }
        // Best-effort CAS — another thread may win, that's fine.
        if self
            .state
            .compare_exchange(prev, CLOSED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.failure_count.store(0, Ordering::Release);
            self.opened_at_ms.store(-1, Ordering::Release);
        }
    }

    /// Call after a failed upstream response (timeout, connection error, 5xx).
    pub fn on_failure(&self) {
        let state = self.state.load(Ordering::Acquire);

        if state == HALF_OPEN {
            // Probe failed — reopen immediately.
            self.opened_at_ms.store(Self::now_ms(), Ordering::Release);
            if self
                .state
                .compare_exchange(HALF_OPEN, OPEN, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                warn!(state = "OPEN", "circuit breaker reopened after failed probe");
            }
            return;
        }

        if state == CLOSED {
            let count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
            warn!(
                failures = count,
                threshold = self.failure_threshold,
                "circuit breaker failure recorded"
            );
            if count >= self.failure_threshold {
                self.opened_at_ms.store(Self::now_ms(), Ordering::Release);
                if self
                    .state
                    .compare_exchange(CLOSED, OPEN, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    warn!(state = "OPEN", "circuit breaker opened");
                }
            }
        }
    }
}
