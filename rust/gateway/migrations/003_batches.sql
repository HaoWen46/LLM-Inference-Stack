CREATE TABLE batches (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash        TEXT        NOT NULL,
    -- validating | in_progress | completed | failed | cancelled
    status          TEXT        NOT NULL DEFAULT 'validating',
    total_requests  INT         NOT NULL DEFAULT 0,
    completed_count INT         NOT NULL DEFAULT 0,
    failed_count    INT         NOT NULL DEFAULT 0,
    requests        TEXT        NOT NULL,        -- JSON-encoded Vec<BatchItem>
    results         TEXT        NOT NULL DEFAULT '[]',  -- JSON-encoded Vec<BatchResult>
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_batches_key_hash ON batches (key_hash);
CREATE INDEX idx_batches_status   ON batches (status);
