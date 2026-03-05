/// PostgreSQL-backed API key store with DashMap in-memory cache.
///
/// Design:
///   - Hot path: DashMap lookup (O(1), no await, no allocations beyond clone)
///   - Admin ops: touch PostgreSQL, then update/invalidate cache atomically
///   - Background task refreshes cache from DB every 60 seconds
///   - Keys stored as SHA-256 hex hashes — plaintext never persisted
use anyhow::Result;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use rand::Rng;
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{error, info};
use uuid::Uuid;

#[derive(Clone)]
pub struct CachedKey {
    pub id: Uuid,
    pub label: String,
    pub expires_at: Option<DateTime<Utc>>,
}

pub struct KeyEntry {
    pub id: Uuid,
    pub key_hash: String,
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub enabled: bool,
    pub last_used_at: Option<DateTime<Utc>>,
}

pub struct KeyStore {
    pool: PgPool,
    /// key_hash (hex) → metadata; populated from DB, updated on admin ops.
    cache: DashMap<String, CachedKey>,
}

impl KeyStore {
    /// Run migrations, optionally seed from `GATEWAY_API_KEYS`, load cache.
    pub async fn init(pool: PgPool, seed_keys: Vec<String>) -> Result<Arc<Self>> {
        sqlx::migrate!("migrations").run(&pool).await?;

        let store = Arc::new(KeyStore {
            pool: pool.clone(),
            cache: DashMap::new(),
        });

        // Seed from env var if DB is empty
        if !seed_keys.is_empty() {
            let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM api_keys")
                .fetch_one(&pool)
                .await?;

            if count == 0 {
                for (i, key) in seed_keys.iter().enumerate() {
                    let hash = sha256_hex(key);
                    let label = format!("migrated-{}", i);
                    sqlx::query(
                        "INSERT INTO api_keys (key_hash, label) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    )
                    .bind(&hash)
                    .bind(&label)
                    .execute(&pool)
                    .await?;
                    info!(label = %label, "seeded API key from GATEWAY_API_KEYS");
                }
            }
        }

        store.reload_cache().await?;
        Ok(store)
    }

    async fn reload_cache(&self) -> Result<()> {
        let rows: Vec<(Uuid, String, String, Option<DateTime<Utc>>)> = sqlx::query_as(
            "SELECT id, key_hash, label, expires_at FROM api_keys WHERE enabled = true",
        )
        .fetch_all(&self.pool)
        .await?;

        self.cache.clear();
        let count = rows.len();
        for (id, key_hash, label, expires_at) in rows {
            self.cache.insert(key_hash, CachedKey { id, label, expires_at });
        }
        info!(keys = count, "key cache reloaded");
        Ok(())
    }

    /// Hot path — pure DashMap lookup, O(1), no await.
    pub fn lookup(&self, key_hash: &str) -> Option<CachedKey> {
        self.cache.get(key_hash).map(|e| e.clone())
    }

    /// Create a new key. Returns `(id, created_at, plaintext_key)`.
    /// The plaintext is shown exactly once and never stored.
    pub async fn create(&self, label: &str) -> Result<(Uuid, DateTime<Utc>, String)> {
        let plaintext = generate_key();
        let hash = sha256_hex(&plaintext);

        let (id, created_at): (Uuid, DateTime<Utc>) = sqlx::query_as(
            "INSERT INTO api_keys (key_hash, label) VALUES ($1, $2) RETURNING id, created_at",
        )
        .bind(&hash)
        .bind(label)
        .fetch_one(&self.pool)
        .await?;

        self.cache.insert(hash, CachedKey { id, label: label.to_string(), expires_at: None });
        Ok((id, created_at, plaintext))
    }

    pub async fn list(&self) -> Result<Vec<KeyEntry>> {
        let rows: Vec<(Uuid, String, String, DateTime<Utc>, Option<DateTime<Utc>>, bool, Option<DateTime<Utc>>)> =
            sqlx::query_as(
                "SELECT id, key_hash, label, created_at, expires_at, enabled, last_used_at \
                 FROM api_keys ORDER BY created_at DESC",
            )
            .fetch_all(&self.pool)
            .await?;

        Ok(rows
            .into_iter()
            .map(|(id, key_hash, label, created_at, expires_at, enabled, last_used_at)| KeyEntry {
                id,
                key_hash,
                label,
                created_at,
                expires_at,
                enabled,
                last_used_at,
            })
            .collect())
    }

    /// Enable or disable a key. Returns `false` if the id was not found.
    pub async fn set_enabled(&self, id: Uuid, enabled: bool) -> Result<bool> {
        let result = sqlx::query("UPDATE api_keys SET enabled = $1 WHERE id = $2")
            .bind(enabled)
            .bind(id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            return Ok(false);
        }
        self.reload_cache().await?;
        Ok(true)
    }

    /// Delete a key permanently. Returns `false` if the id was not found.
    pub async fn delete(&self, id: Uuid) -> Result<bool> {
        let row: Option<(String,)> = sqlx::query_as("SELECT key_hash FROM api_keys WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;

        let result = sqlx::query("DELETE FROM api_keys WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            return Ok(false);
        }
        if let Some((hash,)) = row {
            self.cache.remove(&hash);
        }
        Ok(true)
    }

    /// Fire-and-forget: update `last_used_at` in the background (non-blocking hot path).
    pub fn touch_last_used(&self, id: Uuid) {
        let pool = self.pool.clone();
        tokio::spawn(async move {
            if let Err(e) =
                sqlx::query("UPDATE api_keys SET last_used_at = NOW() WHERE id = $1")
                    .bind(id)
                    .execute(&pool)
                    .await
            {
                error!(error = %e, "touch_last_used failed");
            }
        });
    }

    /// Spawn a background task that reloads the cache from DB every 60 seconds.
    pub fn start_refresh_task(self: &Arc<Self>) {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(60)).await;
                if let Err(e) = store.reload_cache().await {
                    error!(error = %e, "key cache refresh failed");
                }
            }
        });
    }
}

// ── Key generation ────────────────────────────────────────────────────────────

fn generate_key() -> String {
    let bytes: [u8; 32] = rand::thread_rng().gen();
    format!("gw_{}", base64url_encode(&bytes))
}

fn base64url_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len() * 4 / 3 + 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &byte in bytes {
        buf = (buf << 8) | byte as u32;
        bits += 8;
        while bits >= 6 {
            bits -= 6;
            out.push(CHARS[((buf >> bits) & 0x3f) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(CHARS[((buf << (6 - bits)) & 0x3f) as usize] as char);
    }
    out
}

// ── Hashing ───────────────────────────────────────────────────────────────────

pub fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let result = h.finalize();
    let mut out = String::with_capacity(64);
    use std::fmt::Write as _;
    for b in result {
        write!(out, "{:02x}", b).unwrap();
    }
    out
}
