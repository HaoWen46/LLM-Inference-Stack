CREATE TABLE api_keys (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash     TEXT        UNIQUE NOT NULL,
    label        TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ,
    enabled      BOOLEAN     NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMPTZ
);
CREATE INDEX idx_api_keys_hash    ON api_keys (key_hash);
CREATE INDEX idx_api_keys_enabled ON api_keys (enabled);
