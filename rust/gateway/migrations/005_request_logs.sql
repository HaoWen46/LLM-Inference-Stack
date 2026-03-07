-- Request log: one row per completed LLM request.
-- Written asynchronously by a background worker in request_log.rs.
CREATE TABLE request_logs (
    id                BIGSERIAL    PRIMARY KEY,
    key_id            UUID         REFERENCES api_keys(id) ON DELETE SET NULL,
    key_hash          TEXT         NOT NULL,
    model             TEXT         NOT NULL,
    endpoint          TEXT         NOT NULL,
    prompt_tokens     BIGINT       NOT NULL DEFAULT 0,
    completion_tokens BIGINT       NOT NULL DEFAULT 0,
    latency_ms        INTEGER      NOT NULL DEFAULT 0,
    status_code       SMALLINT     NOT NULL DEFAULT 200,
    truncated_prompt  TEXT,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_request_logs_key_id   ON request_logs (key_id);
CREATE INDEX idx_request_logs_key_hash ON request_logs (key_hash);
CREATE INDEX idx_request_logs_model    ON request_logs (model);
CREATE INDEX idx_request_logs_created  ON request_logs (created_at DESC);
