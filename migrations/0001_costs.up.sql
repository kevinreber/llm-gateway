-- Every completion the gateway proxies produces one row here.
--
-- cost_cents is NUMERIC rather than a float: this is money, and it gets
-- SUMmed for billing reports. NUMERIC(12,6) holds six decimal places
-- because a single Haiku request can cost well under a hundredth of a
-- cent, and rounding those to zero would make per-request accounting
-- useless at low volume.
CREATE TABLE costs (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    provider    TEXT         NOT NULL,
    model       TEXT         NOT NULL,
    alias       TEXT,
    input_tok   INTEGER      NOT NULL,
    output_tok  INTEGER      NOT NULL,
    cost_cents  NUMERIC(12,6) NOT NULL
);

-- Cost queries are always time-bounded ("spend this month"), and usually
-- also grouped by provider or alias.
CREATE INDEX idx_costs_ts ON costs (ts);
CREATE INDEX idx_costs_provider_ts ON costs (provider, ts);
CREATE INDEX idx_costs_alias_ts ON costs (alias, ts) WHERE alias IS NOT NULL;
