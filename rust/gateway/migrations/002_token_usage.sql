CREATE TABLE token_usage (
    key_hash          TEXT   NOT NULL,
    date              DATE   NOT NULL,
    prompt_tokens     BIGINT NOT NULL DEFAULT 0,
    completion_tokens BIGINT NOT NULL DEFAULT 0,
    request_count     BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (key_hash, date)
);
