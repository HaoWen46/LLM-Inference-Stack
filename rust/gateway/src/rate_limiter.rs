use dashmap::DashMap;
use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::{net::IpAddr, num::NonZeroU32, sync::Arc};

pub type SingleLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

/// Per-IP GCRA rate limiter backed by a DashMap.
/// A new limiter is created lazily on first request from each IP.
pub struct RateLimiterMap {
    limiters: DashMap<IpAddr, Arc<SingleLimiter>>,
    quota: Quota,
}

impl RateLimiterMap {
    pub fn new(requests_per_minute: u32) -> Self {
        let rpm = NonZeroU32::new(requests_per_minute).unwrap_or(NonZeroU32::new(60).unwrap());
        let quota = Quota::per_minute(rpm);
        Self {
            limiters: DashMap::new(),
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
        limiter.check().is_ok()
    }
}
