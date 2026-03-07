/// Non-blocking request logger backed by PostgreSQL.
///
/// The hot path calls `RequestLogger::log()` which enqueues a `LogEntry` into
/// a bounded `mpsc` channel.  A background task drains the channel and inserts
/// rows into `request_logs` one by one.  If the channel is full the entry is
/// silently dropped — we prefer losing a log line over slowing down the proxy.
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, warn};
use uuid::Uuid;

// ── Public types ──────────────────────────────────────────────────────────────

pub struct LogEntry {
    pub key_id: Option<Uuid>,
    pub key_hash: String,
    pub model: String,
    pub endpoint: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub latency_ms: u32,
    pub status_code: u16,
    pub truncated_prompt: Option<String>,
}

pub struct RequestLogger {
    sender: mpsc::Sender<LogEntry>,
}

impl RequestLogger {
    /// Spawn the background worker and return a handle.
    pub fn new(pool: PgPool) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<LogEntry>(8192);
        tokio::spawn(log_worker(pool, rx));
        Arc::new(Self { sender: tx })
    }

    /// Enqueue a log entry.  Non-blocking — drops the entry if the channel is full.
    pub fn log(&self, entry: LogEntry) {
        match self.sender.try_send(entry) {
            Ok(_) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("request_log channel full — dropping log entry");
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {}
        }
    }
}

// ── Background worker ─────────────────────────────────────────────────────────

async fn log_worker(pool: PgPool, mut rx: mpsc::Receiver<LogEntry>) {
    while let Some(e) = rx.recv().await {
        if let Err(err) = sqlx::query(
            "INSERT INTO request_logs \
             (key_id, key_hash, model, endpoint, \
              prompt_tokens, completion_tokens, latency_ms, status_code, truncated_prompt) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(e.key_id)
        .bind(&e.key_hash)
        .bind(&e.model)
        .bind(&e.endpoint)
        .bind(e.prompt_tokens as i64)
        .bind(e.completion_tokens as i64)
        .bind(e.latency_ms as i32)
        .bind(e.status_code as i16)
        .bind(&e.truncated_prompt)
        .execute(&pool)
        .await
        {
            error!(error = %err, "request_log: failed to insert row");
        }
    }
}
