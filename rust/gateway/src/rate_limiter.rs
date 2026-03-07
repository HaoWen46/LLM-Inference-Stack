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

/// Per-IP and per-key GCRA rate limiters.
///
/// Per-IP: new limiter created lazily on first request from each IP.
/// Per-key: new limiter created lazily using the key's configured RPM; recreated
///   automatically if the RPM setting changes (detected on each check).
/// Background cleanup task evicts entries idle for longer than `IDLE_TTL`.
pub struct RateLimiterMap {
    // Per-IP
    limiters: DashMap<IpAddr, Arc<SingleLimiter>>,
    last_seen: DashMap<IpAddr, Instant>,
    quota: Quota,
    // Per-key (key_hash → (limiter, configured_rpm))
    key_limiters: DashMap<String, (Arc<SingleLimiter>, u32)>,
    key_last_seen: DashMap<String, Instant>,
}

/// Evict per-IP/key state that hasn't been used in this long.
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
            key_limiters: DashMap::new(),
            key_last_seen: DashMap::new(),
        }
    }

    /// Check per-IP rate limit. Returns `true` if the request is allowed.
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

    /// Check per-key RPM rate limit. Returns `true` if the request is allowed.
    /// Creates or recreates the limiter if `rpm` has changed since last call.
    pub fn check_key(&self, key_hash: &str, rpm: u32) -> bool {
        let rpm_nz = NonZeroU32::new(rpm).unwrap_or(NonZeroU32::new(1).unwrap());
        let limiter = {
            let mut entry = self
                .key_limiters
                .entry(key_hash.to_string())
                .or_insert_with(|| {
                    (Arc::new(RateLimiter::direct(Quota::per_minute(rpm_nz))), rpm)
                });
            // Recreate if RPM limit was changed via admin API
            if entry.1 != rpm {
                *entry = (Arc::new(RateLimiter::direct(Quota::per_minute(rpm_nz))), rpm);
            }
            Arc::clone(&entry.0)
        };
        self.key_last_seen.insert(key_hash.to_string(), Instant::now());
        limiter.check().is_ok()
    }

    /// Evict limiters not seen in the last `IDLE_TTL`. Called by the cleanup task.
    fn evict_idle(&self) {
        let now = Instant::now();

        // Per-IP eviction
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
            debug!(evicted, remaining = self.limiters.len(), "per-IP rate limiter eviction");
        }

        // Per-key eviction
        let before_key = self.key_limiters.len();
        self.key_last_seen.retain(|hash, last_seen| {
            let keep = now.duration_since(*last_seen) <= IDLE_TTL;
            if !keep {
                self.key_limiters.remove(hash);
            }
            keep
        });
        let evicted_key = before_key.saturating_sub(self.key_limiters.len());
        if evicted_key > 0 {
            debug!(evicted_key, remaining = self.key_limiters.len(), "per-key rate limiter eviction");
        }
    }

    /// Spawn a background task that periodically evicts idle per-IP/key state,
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
