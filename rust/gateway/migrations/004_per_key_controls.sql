-- Milestone 2: Per-key controls
-- Adds organisations table and extends api_keys with per-key rate limits,
-- daily token limit override, model allowlist, and org membership.

CREATE TABLE orgs (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT        NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE api_keys
    ADD COLUMN IF NOT EXISTS rpm_limit          INT,           -- NULL = inherit global rate limit
    ADD COLUMN IF NOT EXISTS daily_token_limit  BIGINT,        -- NULL = inherit global DAILY_TOKEN_QUOTA
    ADD COLUMN IF NOT EXISTS allowed_models     TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS org_id             UUID REFERENCES orgs(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys (org_id) WHERE org_id IS NOT NULL;
