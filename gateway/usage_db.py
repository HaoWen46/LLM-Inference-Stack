"""
Per-API-key daily token quota tracking using SQLite (via aiosqlite).

Keys are stored as SHA-256 hashes — plaintext never touches the DB.
The quota window resets at UTC midnight (date-keyed rows).

Schema:
    api_key_usage(
        key_hash      TEXT,
        date          TEXT,       -- YYYY-MM-DD (UTC)
        prompt_tokens INT,
        completion_tokens INT,
        request_count INT,
        PRIMARY KEY (key_hash, date)
    )
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import structlog

from gateway.config import settings

log = structlog.get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "usage.db"

_init_lock = asyncio.Lock()
_initialized = False


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


async def _get_db() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    return db


async def init_db() -> None:
    """Create schema on first run. Safe to call multiple times."""
    global _initialized
    async with _init_lock:
        if _initialized:
            return
        async with await _get_db() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS api_key_usage (
                    key_hash          TEXT NOT NULL,
                    date              TEXT NOT NULL,
                    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    request_count     INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (key_hash, date)
                )
            """)
            await db.commit()
        _initialized = True
        log.info("usage_db_initialized", path=str(DB_PATH))


async def record_usage(api_key: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Upsert token usage for the given key on today's date."""
    key_hash = _hash_key(api_key)
    today = _today()
    async with await _get_db() as db:
        await db.execute(
            """
            INSERT INTO api_key_usage (key_hash, date, prompt_tokens, completion_tokens, request_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(key_hash, date) DO UPDATE SET
                prompt_tokens     = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens,
                request_count     = request_count + 1
            """,
            (key_hash, today, prompt_tokens, completion_tokens),
        )
        await db.commit()


async def get_usage(api_key: str, date: str | None = None) -> dict:
    """Return usage stats for the given key on the given date (default: today)."""
    key_hash = _hash_key(api_key)
    date = date or _today()
    async with await _get_db() as db:
        async with db.execute(
            "SELECT prompt_tokens, completion_tokens, request_count FROM api_key_usage WHERE key_hash=? AND date=?",
            (key_hash, date),
        ) as cursor:
            row = await cursor.fetchone()
    if row:
        return {
            "date": date,
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "request_count": row["request_count"],
        }
    return {"date": date, "prompt_tokens": 0, "completion_tokens": 0, "request_count": 0}


async def check_quota(api_key: str) -> dict:
    """
    Check whether the key has remaining quota for today.

    Returns:
        {
            "allowed": bool,
            "quota": int,           # configured limit (0 = unlimited)
            "used_tokens": int,     # total tokens used today
            "remaining": int,       # remaining tokens (-1 if unlimited)
        }
    """
    quota = settings.daily_token_quota
    if quota <= 0:
        return {"allowed": True, "quota": 0, "used_tokens": 0, "remaining": -1}

    usage = await get_usage(api_key)
    used = usage["prompt_tokens"] + usage["completion_tokens"]
    remaining = max(0, quota - used)
    return {
        "allowed": remaining > 0,
        "quota": quota,
        "used_tokens": used,
        "remaining": remaining,
    }
