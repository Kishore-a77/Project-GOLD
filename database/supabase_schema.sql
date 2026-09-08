-- database/supabase_schema.sql
--
-- Save this file at exactly this path in your repo: database/supabase_schema.sql
-- `app/db/supabase_client.py::ensure_schema()` looks for it there and runs it
-- automatically (via DATABASE_URL) on every pipeline run, so once this file
-- exists, STAGE 0 will stop warning "Schema file not found" and these tables
-- will always exist before the pipeline touches them -- including on a fresh
-- Supabase project or in CI.
--
-- Safe to run multiple times: every statement is idempotent (IF NOT EXISTS).

-- Already created manually, included here for completeness / fresh installs.
CREATE TABLE IF NOT EXISTS gold_prices (
    date    DATE PRIMARY KEY,
    open    DOUBLE PRECISION,
    high    DOUBLE PRECISION,
    low     DOUBLE PRECISION,
    close   DOUBLE PRECISION,
    volume  BIGINT
);

-- Written by services/feature_service.py::upsert_features().
-- upsert() is called with no on_conflict, so PostgREST upserts against the
-- primary key -- `date` must be PRIMARY KEY for that to work correctly.
CREATE TABLE IF NOT EXISTS gold_features (
    date         DATE PRIMARY KEY,
    close        DOUBLE PRECISION,
    sma_7        DOUBLE PRECISION,
    sma_30       DOUBLE PRECISION,
    rsi_14       DOUBLE PRECISION,
    macd         DOUBLE PRECISION,
    macd_signal  DOUBLE PRECISION,
    bb_upper     DOUBLE PRECISION,
    bb_lower     DOUBLE PRECISION,
    atr_14       DOUBLE PRECISION
);

-- Written by services/prediction_service.py::save_predictions().
-- Comment in that file says "idempotent upsert on (date, horizon)" --
-- that's the composite primary key PostgREST needs for the on-conflict-free
-- upsert() call to behave correctly.
CREATE TABLE IF NOT EXISTS predictions (
    date           DATE NOT NULL,
    horizon        TEXT NOT NULL,   -- '1d' | '7d' | '30d' | '90d' | '180d' | '365d'
    chronos_pred   DOUBLE PRECISION,
    nhits_pred     DOUBLE PRECISION,
    ensemble_pred  DOUBLE PRECISION,
    model_version  TEXT,
    PRIMARY KEY (date, horizon)
);

-- Written by services/prediction_service.py::record_run_status().
-- Also called via .upsert(row) with no on_conflict -- one status row per
-- calendar day (a second run the same day overwrites the first).
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_date               DATE PRIMARY KEY,
    started_at             TIMESTAMPTZ,
    finished_at            TIMESTAMPTZ,
    status                 TEXT,     -- 'success' | 'failed'
    error                  TEXT,
    model_version          TEXT,
    records_processed      INTEGER,
    predictions_generated  INTEGER
);

-- Written by automation/weekly_training.py and read by the dashboard.
-- One row represents a promoted model version; model_name + version is
-- unique so repeated metadata writes remain identifiable.
CREATE TABLE IF NOT EXISTS model_metadata (
    id            BIGSERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    mae           DOUBLE PRECISION,
    rmse          DOUBLE PRECISION,
    mape          DOUBLE PRECISION,
    artifact_path TEXT,
    is_active     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
);

CREATE INDEX IF NOT EXISTS model_metadata_active_idx
    ON model_metadata (model_name, is_active);
