use dashmap::DashMap;
use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::{net::IpAddr, num::NonZeroU32, sync::Arc, time::Instant};
use tokio::time::{sleep, Duration};
use tracing::debug;

pub type SingleLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

/// Per-IP GCRA rate limiter backed by a DashMap.
/// A new limiter is created lazily on first request from each IP.
/// A background cleanup task evicts entries idle for longer than `IDLE_TTL`.
pub struct RateLimiterMap {
    limiters: DashMap<IpAddr, Arc<SingleLimiter>>,
    /// Last time each IP was seen — used for TTL eviction.
    last_seen: DashMap<IpAddr, Instant>,
    quota: Quota,
}

/// Evict per-IP state that hasn't been used in this long.
const IDLE_TTL: Duration = Duration::from_secs(3600); // 1 hour
/// How often to run the eviction sweep.
const CLEANUP_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes

impl RateLimiterMap {
    pub fn new(requests_per_minute: u32) -> Self {
        let rpm = NonZeroU32::new(requests_per_minute).unwrap_or(NonZeroU32::new(60).unwrap());
        let quota = Quota::per_minute(rpm);
        Self {
            limiters: DashMap::new(),
            last_seen: DashMap::new(),
            quota,
        }
    }

    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn check(&self, ip: IpAddr) -> bool {
        let limiter = {
            let entry = self
                .limiters
                .entry(ip)
                .or_insert_with(|| Arc::new(RateLimiter::direct(self.quota)));
            Arc::clone(&*entry)
        };
        self.last_seen.insert(ip, Instant::now());
        limiter.check().is_ok()
    }

    /// Evict limiters not seen in the last `IDLE_TTL`. Called by the cleanup task.
    fn evict_idle(&self) {
        let now = Instant::now();
        let before = self.limiters.len();
        self.last_seen.retain(|ip, last_seen| {
            let keep = now.duration_since(*last_seen) <= IDLE_TTL;
            if !keep {
                self.limiters.remove(ip);
            }
            keep
        });
        let evicted = before.saturating_sub(self.limiters.len());
        if evicted > 0 {
            debug!(evicted, remaining = self.limiters.len(), "rate limiter eviction");
        }
    }

    /// Spawn a background task that periodically evicts idle per-IP state,
    /// preventing unbounded memory growth from unique-IP attacks.
    pub fn start_cleanup_task(self: &Arc<Self>) {
        let map = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                sleep(CLEANUP_INTERVAL).await;
                map.evict_idle();
            }
        });
    }
}
