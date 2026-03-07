/// PostgreSQL-backed API key store with DashMap in-memory cache.
///
/// Design:
///   - Hot path: DashMap lookup (O(1), no await, no allocations beyond clone)
///   - Admin ops: touch PostgreSQL, then update/invalidate cache atomically
///   - Background task refreshes cache from DB every 60 seconds
///   - Keys stored as SHA-256 hashes — plaintext never persisted
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

// ── Cached key (hot-path struct) ──────────────────────────────────────────────

#[derive(Clone)]
pub struct CachedKey {
    pub id: Uuid,
    pub label: String,
    pub expires_at: Option<DateTime<Utc>>,
    /// Per-key RPM limit — None means use global RATE_LIMIT_PER_MINUTE.
    pub rpm_limit: Option<i32>,
    /// Per-key daily token limit — None means use global DAILY_TOKEN_QUOTA.
    pub daily_token_limit: Option<i64>,
    /// Allowed model IDs — empty means all models allowed.
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
}

// ── Full key entry (returned by list()) ───────────────────────────────────────

pub struct KeyEntry {
    pub id: Uuid,
    pub key_hash: String,
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub enabled: bool,
    pub last_used_at: Option<DateTime<Utc>>,
    pub rpm_limit: Option<i32>,
    pub daily_token_limit: Option<i64>,
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
}

// ── Options for creating / updating keys ─────────────────────────────────────

pub struct CreateKeyOptions {
    pub label: String,
    pub expires_at: Option<DateTime<Utc>>,
    pub rpm_limit: Option<i32>,
    pub daily_token_limit: Option<i64>,
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
}

/// PUT /admin/keys/:id/limits — replace all limit fields (PUT semantics).
pub struct KeyLimits {
    pub expires_at: Option<DateTime<Utc>>,
    pub rpm_limit: Option<i32>,
    pub daily_token_limit: Option<i64>,
    pub allowed_models: Vec<String>,
    pub org_id: Option<Uuid>,
}

// ── Org types ─────────────────────────────────────────────────────────────────

pub struct OrgEntry {
    pub id: Uuid,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

pub struct OrgUsage {
    pub date: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub request_count: i64,
    pub key_count: i64,
}

// ── KeyStore ──────────────────────────────────────────────────────────────────

pub struct KeyStore {
    pool: PgPool,
    /// key_hash (hex) → metadata; populated from DB, updated on admin ops.
    cache: DashMap<String, CachedKey>,
}

impl KeyStore {
    /// Run migrations, optionally seed from `GATEWAY_API_KEYS`, load cache.
    pub async fn init(pool: PgPool, seed_keys: Vec<String>) -> Result<Arc<Self>> {
        sqlx::migrate!().run(&pool).await?;

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
        let rows: Vec<(Uuid, String, String, Option<DateTime<Utc>>, Option<i32>, Option<i64>, Vec<String>, Option<Uuid>)> =
            sqlx::query_as(
                "SELECT id, key_hash, label, expires_at, rpm_limit, daily_token_limit, \
                        allowed_models, org_id \
                 FROM api_keys WHERE enabled = true",
            )
            .fetch_all(&self.pool)
            .await?;

        self.cache.clear();
        let count = rows.len();
        for (id, key_hash, label, expires_at, rpm_limit, daily_token_limit, allowed_models, org_id) in rows {
            self.cache.insert(
                key_hash,
                CachedKey { id, label, expires_at, rpm_limit, daily_token_limit, allowed_models, org_id },
            );
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
    pub async fn create(&self, opts: CreateKeyOptions) -> Result<(Uuid, DateTime<Utc>, String)> {
        let plaintext = generate_key();
        let hash = sha256_hex(&plaintext);

        let (id, created_at): (Uuid, DateTime<Utc>) = sqlx::query_as(
            "INSERT INTO api_keys \
             (key_hash, label, expires_at, rpm_limit, daily_token_limit, allowed_models, org_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             RETURNING id, created_at",
        )
        .bind(&hash)
        .bind(&opts.label)
        .bind(opts.expires_at)
        .bind(opts.rpm_limit)
        .bind(opts.daily_token_limit)
        .bind(&opts.allowed_models)
        .bind(opts.org_id)
        .fetch_one(&self.pool)
        .await?;

        self.cache.insert(
            hash,
            CachedKey {
                id,
                label: opts.label,
                expires_at: opts.expires_at,
                rpm_limit: opts.rpm_limit,
                daily_token_limit: opts.daily_token_limit,
                allowed_models: opts.allowed_models,
                org_id: opts.org_id,
            },
        );
        Ok((id, created_at, plaintext))
    }

    pub async fn list(&self) -> Result<Vec<KeyEntry>> {
        let rows: Vec<(
            Uuid, String, String, DateTime<Utc>,
            Option<DateTime<Utc>>, bool, Option<DateTime<Utc>>,
            Option<i32>, Option<i64>, Vec<String>, Option<Uuid>,
        )> = sqlx::query_as(
            "SELECT id, key_hash, label, created_at, expires_at, enabled, last_used_at, \
                    rpm_limit, daily_token_limit, allowed_models, org_id \
             FROM api_keys ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(id, key_hash, label, created_at, expires_at, enabled, last_used_at,
                  rpm_limit, daily_token_limit, allowed_models, org_id)| KeyEntry {
                    id,
                    key_hash,
                    label,
                    created_at,
                    expires_at,
                    enabled,
                    last_used_at,
                    rpm_limit,
                    daily_token_limit,
                    allowed_models,
                    org_id,
                },
            )
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

    /// Update per-key limits (PUT semantics — replaces all limit fields).
    /// Returns `false` if the id was not found.
    pub async fn update_limits(&self, id: Uuid, limits: KeyLimits) -> Result<bool> {
        let row: Option<(String,)> = sqlx::query_as(
            "UPDATE api_keys \
             SET expires_at = $2, rpm_limit = $3, daily_token_limit = $4, \
                 allowed_models = $5, org_id = $6 \
             WHERE id = $1 \
             RETURNING key_hash",
        )
        .bind(id)
        .bind(limits.expires_at)
        .bind(limits.rpm_limit)
        .bind(limits.daily_token_limit)
        .bind(&limits.allowed_models)
        .bind(limits.org_id)
        .fetch_optional(&self.pool)
        .await?;

        let key_hash = match row {
            Some((h,)) => h,
            None => return Ok(false),
        };

        // Update in-place to avoid a full reload
        if let Some(mut entry) = self.cache.get_mut(&key_hash) {
            entry.expires_at = limits.expires_at;
            entry.rpm_limit = limits.rpm_limit;
            entry.daily_token_limit = limits.daily_token_limit;
            entry.allowed_models = limits.allowed_models;
            entry.org_id = limits.org_id;
        }
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

    // ── Org operations ────────────────────────────────────────────────────────

    pub async fn create_org(&self, name: &str) -> Result<OrgEntry> {
        let (id, created_at): (Uuid, DateTime<Utc>) = sqlx::query_as(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id, created_at",
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await?;
        Ok(OrgEntry { id, name: name.to_string(), created_at })
    }

    pub async fn list_orgs(&self) -> Result<Vec<OrgEntry>> {
        let rows: Vec<(Uuid, String, DateTime<Utc>)> =
            sqlx::query_as("SELECT id, name, created_at FROM orgs ORDER BY created_at DESC")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .into_iter()
            .map(|(id, name, created_at)| OrgEntry { id, name, created_at })
            .collect())
    }

    /// Aggregate token usage for all keys belonging to `org_id` on `date`.
    pub async fn get_org_usage(&self, org_id: Uuid, date: &str) -> Result<OrgUsage> {
        let row: Option<(i64, i64, i64, i64)> = sqlx::query_as(
            "SELECT \
                 COALESCE(SUM(tu.prompt_tokens), 0)::BIGINT, \
                 COALESCE(SUM(tu.completion_tokens), 0)::BIGINT, \
                 COALESCE(SUM(tu.request_count), 0)::BIGINT, \
                 COUNT(DISTINCT ak.id)::BIGINT \
             FROM api_keys ak \
             LEFT JOIN token_usage tu \
                 ON tu.key_hash = ak.key_hash AND tu.date = $2::date \
             WHERE ak.org_id = $1",
        )
        .bind(org_id)
        .bind(date)
        .fetch_optional(&self.pool)
        .await?;

        Ok(match row {
            Some((prompt, completion, reqs, keys)) => OrgUsage {
                date: date.to_string(),
                prompt_tokens: prompt,
                completion_tokens: completion,
                request_count: reqs,
                key_count: keys,
            },
            None => OrgUsage {
                date: date.to_string(),
                prompt_tokens: 0,
                completion_tokens: 0,
                request_count: 0,
                key_count: 0,
            },
        })
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
