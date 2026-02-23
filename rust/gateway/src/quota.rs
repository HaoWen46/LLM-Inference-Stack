/// Per-API-key daily token quota tracking.
///
/// Design:
///   - In-memory DashMap cache for fast (~sub-µs) quota checks
///   - SQLite backend for persistence across restarts
///   - Background tokio task flushes cache → SQLite every 10 seconds
///   - Midnight UTC reset detected by background task
///
/// Keys are stored as SHA-256 hashes — plaintext never touches the DB.
use anyhow::Result;
use chrono::Utc;
use dashmap::DashMap;
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{error, info};

type KeyHash = [u8; 32];

struct CacheEntry {
    date: String,
    /// Combined (prompt + completion) token count for today.
    tokens: u64,
}

pub struct QuotaStore {
    cache: Arc<DashMap<KeyHash, CacheEntry>>,
    pool: SqlitePool,
    pub daily_quota: u64,
}

impl QuotaStore {
    /// Initialize the store: create the SQLite schema and load today's rows.
    pub async fn new(daily_quota: u64, db_path: &str) -> Result<Arc<Self>> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let url = format!("sqlite://{}?mode=rwc", db_path);
        let pool = SqlitePool::connect(&url).await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS api_key_usage (
                key_hash          TEXT NOT NULL,
                date              TEXT NOT NULL,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                request_count     INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key_hash, date)
            )",
        )
        .execute(&pool)
        .await?;

        info!(path = db_path, "quota db initialized");

        let cache = Arc::new(DashMap::new());
        let store = Arc::new(QuotaStore {
            cache: cache.clone(),
            pool,
            daily_quota,
        });

        // Pre-load today's rows into the cache
        store.load_today().await?;

        Ok(store)
    }

    /// Load today's usage rows from SQLite into the in-memory cache.
    async fn load_today(&self) -> Result<()> {
        let today = today();
        let rows = sqlx::query_as::<_, (String, i64, i64)>(
            "SELECT key_hash, prompt_tokens, completion_tokens
             FROM api_key_usage WHERE date = ?",
        )
        .bind(&today)
        .fetch_all(&self.pool)
        .await?;

        let row_count = rows.len();
        for (key_hex, prompt, completion) in rows {
            if let Some(hash) = hex_to_hash(&key_hex) {
                self.cache.insert(
                    hash,
                    CacheEntry {
                        date: today.clone(),
                        tokens: (prompt + completion).max(0) as u64,
                    },
                );
            }
        }
        info!(date = %today, rows = row_count, "quota cache loaded");
        Ok(())
    }

    /// Check whether the key has remaining quota. Does not charge tokens.
    /// Returns `true` if the request is allowed (includes unlimited case).
    pub fn is_allowed(&self, api_key: &str) -> bool {
        if self.daily_quota == 0 {
            return true;
        }
        let hash = hash_key(api_key);
        let today_str = today();
        match self.cache.get(&hash) {
            Some(entry) if entry.date == today_str => entry.tokens < self.daily_quota,
            // No entry means 0 tokens used today → allowed
            _ => true,
        }
    }

    /// Record token usage. Atomic fetch_add on the in-memory cache; SQLite is
    /// updated by the background flush task.
    pub fn add_tokens(&self, api_key: &str, prompt_tokens: u64, completion_tokens: u64) {
        let hash = hash_key(api_key);
        let today_str = today();
        let delta = prompt_tokens + completion_tokens;

        let mut entry = self
            .cache
            .entry(hash)
            .or_insert_with(|| CacheEntry {
                date: today_str.clone(),
                tokens: 0,
            });

        // Reset if date has changed (crossed midnight while running)
        if entry.date != today_str {
            entry.date = today_str;
            entry.tokens = 0;
        }

        entry.tokens += delta;
    }

    /// Increment request count in SQLite. Called from the background flush.
    pub async fn record_request_db(
        &self,
        key_hash_hex: &str,
        date: &str,
        prompt_tokens: i64,
        completion_tokens: i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO api_key_usage (key_hash, date, prompt_tokens, completion_tokens, request_count)
             VALUES (?, ?, ?, ?, 1)
             ON CONFLICT(key_hash, date) DO UPDATE SET
                 prompt_tokens     = excluded.prompt_tokens,
                 completion_tokens = excluded.completion_tokens,
                 request_count     = request_count + 1",
        )
        .bind(key_hash_hex)
        .bind(date)
        .bind(prompt_tokens)
        .bind(completion_tokens)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Return today's usage for a given API key.
    pub async fn get_usage(&self, api_key: &str) -> Result<UsageInfo> {
        let key_hex = hash_hex(api_key);
        let today_str = today();

        let row = sqlx::query_as::<_, (i64, i64, i64)>(
            "SELECT prompt_tokens, completion_tokens, request_count
             FROM api_key_usage WHERE key_hash = ? AND date = ?",
        )
        .bind(&key_hex)
        .bind(&today_str)
        .fetch_optional(&self.pool)
        .await?;

        Ok(match row {
            Some((prompt, completion, reqs)) => UsageInfo {
                date: today_str,
                prompt_tokens: prompt,
                completion_tokens: completion,
                request_count: reqs,
            },
            None => UsageInfo {
                date: today_str,
                prompt_tokens: 0,
                completion_tokens: 0,
                request_count: 0,
            },
        })
    }

    /// Flush in-memory cache to SQLite and reset date entries for keys that
    /// have crossed midnight. Called by the background task every 10 seconds.
    pub async fn flush(&self) {
        let today_str = today();

        // Collect entries to flush (avoid holding DashMap lock across awaits)
        let entries: Vec<(KeyHash, String, u64)> = self
            .cache
            .iter()
            .map(|e| (*e.key(), e.value().date.clone(), e.value().tokens))
            .collect();

        for (hash, date, tokens) in &entries {
            let key_hex = hex_encode(hash);
            // Split tokens roughly 50/50 for DB schema compatibility;
            // the quota check uses the combined total from the cache.
            let half = (*tokens / 2) as i64;
            let other_half = (*tokens - (*tokens / 2)) as i64;

            if let Err(e) = sqlx::query(
                "INSERT INTO api_key_usage (key_hash, date, prompt_tokens, completion_tokens, request_count)
                 VALUES (?, ?, ?, ?, 0)
                 ON CONFLICT(key_hash, date) DO UPDATE SET
                     prompt_tokens     = excluded.prompt_tokens,
                     completion_tokens = excluded.completion_tokens",
            )
            .bind(&key_hex)
            .bind(date)
            .bind(half)
            .bind(other_half)
            .execute(&self.pool)
            .await
            {
                error!(error = %e, "quota flush error");
            }
        }

        // Evict stale date entries from cache
        self.cache.retain(|_, v| v.date == today_str);
    }

    /// Spawn the background flush + midnight-reset task.
    pub fn start_flush_task(self: &Arc<Self>) {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(10)).await;
                store.flush().await;
            }
        });
    }
}

// ── Usage DTO ────────────────────────────────────────────────────────────────

pub struct UsageInfo {
    pub date: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub request_count: i64,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn hash_key(api_key: &str) -> KeyHash {
    let mut h = Sha256::new();
    h.update(api_key.as_bytes());
    h.finalize().into()
}

fn hash_hex(api_key: &str) -> String {
    hex_encode(&hash_key(api_key))
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    // Pre-allocate exact capacity (32 bytes → 64 hex chars) and write in one pass.
    let mut s = String::with_capacity(64);
    for b in bytes {
        write!(s, "{:02x}", b).unwrap();
    }
    s
}

fn hex_to_hash(s: &str) -> Option<KeyHash> {
    if s.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        bytes[i] = (hi << 4) | lo;
    }
    Some(bytes)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn today() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}
