-- =============================================================================
-- GIC Financial Intelligence — Supabase Seed Data
-- File:    supabase/seed.sql
-- Purpose: Populate demo users, sample audit events, and sample forecasts.
--          Requires pgcrypto (loaded by 001_init.sql).
--          All inserts are idempotent via ON CONFLICT DO NOTHING.
-- =============================================================================

-- Requires: 001_init.sql must have been run first.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- USERS  (2 demo accounts)
-- Passwords are hashed with bcrypt via pgcrypto crypt().
-- Cost factor 10 is standard production default.
-- ---------------------------------------------------------------------------

-- Admin account: username=admin, password=admin123
INSERT INTO users (id, username, password_hash, role, full_name, email)
VALUES (
    'a0000000-0000-0000-0000-000000000001',
    'admin',
    crypt('admin123', gen_salt('bf', 10)),
    'admin',
    'Alex Morgan',
    'admin@gic-intelligence.io'
)
ON CONFLICT (username) DO NOTHING;

-- Analyst account: username=user, password=user123
INSERT INTO users (id, username, password_hash, role, full_name, email)
VALUES (
    'a0000000-0000-0000-0000-000000000002',
    'user',
    crypt('user123', gen_salt('bf', 10)),
    'user',
    'Jordan Lee',
    'analyst@gic-intelligence.io'
)
ON CONFLICT (username) DO NOTHING;

-- ---------------------------------------------------------------------------
-- AUDIT EVENTS  (5 representative events)
-- ---------------------------------------------------------------------------

INSERT INTO audit_events (id, event_type, user_id, details, created_at)
VALUES
    (
        'e0000000-0000-0000-0000-000000000001',
        'login',
        'admin',
        '{"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"}',
        now() - INTERVAL '2 days'
    ),
    (
        'e0000000-0000-0000-0000-000000000002',
        'forecast_run',
        'admin',
        '{"commodity": "Copper", "horizon_months": 12, "model": "xgboost"}',
        now() - INTERVAL '2 days' + INTERVAL '5 minutes'
    ),
    (
        'e0000000-0000-0000-0000-000000000003',
        'scenario_run',
        'admin',
        '{"name": "Supply Crunch Q3", "demand_shock": -0.15, "commodity_shock": 0.25, "fx_shock": 0.05}',
        now() - INTERVAL '1 day'
    ),
    (
        'e0000000-0000-0000-0000-000000000004',
        'login',
        'user',
        '{"ip": "10.0.0.42", "user_agent": "Chrome/120"}',
        now() - INTERVAL '12 hours'
    ),
    (
        'e0000000-0000-0000-0000-000000000005',
        'forecast_run',
        'user',
        '{"commodity": "Lithium", "horizon_months": 6, "model": "ensemble"}',
        now() - INTERVAL '11 hours'
    )
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- FORECASTS  (3 sample runs: Copper, Lithium, Steel)
-- ---------------------------------------------------------------------------

-- Copper — 12-month XGBoost point forecast (monthly, USD/tonne)
INSERT INTO forecasts (id, commodity, model_type, point_forecast, metrics, feature_importance, created_at)
VALUES (
    'f0000000-0000-0000-0000-000000000001',
    'Copper',
    'xgboost',
    '[
        {"date": "2026-01", "value": 9120},
        {"date": "2026-02", "value": 9245},
        {"date": "2026-03", "value": 9380},
        {"date": "2026-04", "value": 9210},
        {"date": "2026-05", "value": 9450},
        {"date": "2026-06", "value": 9610},
        {"date": "2026-07", "value": 9530},
        {"date": "2026-08", "value": 9680},
        {"date": "2026-09", "value": 9820},
        {"date": "2026-10", "value": 9750},
        {"date": "2026-11", "value": 9900},
        {"date": "2026-12", "value": 10050}
    ]',
    '{"mae": 142.3, "rmse": 198.7, "mape": 1.54, "r2": 0.923}',
    '{"lag_1": 0.38, "lag_3": 0.21, "usd_index": 0.17, "china_pmi": 0.14, "inventory": 0.10}',
    now() - INTERVAL '2 days' + INTERVAL '5 minutes'
)
ON CONFLICT (id) DO NOTHING;

-- Lithium — 6-month Ensemble point forecast (monthly, USD/tonne)
INSERT INTO forecasts (id, commodity, model_type, point_forecast, metrics, feature_importance, created_at)
VALUES (
    'f0000000-0000-0000-0000-000000000002',
    'Lithium',
    'ensemble',
    '[
        {"date": "2026-01", "value": 14800},
        {"date": "2026-02", "value": 14650},
        {"date": "2026-03", "value": 14400},
        {"date": "2026-04", "value": 14200},
        {"date": "2026-05", "value": 14050},
        {"date": "2026-06", "value": 13900}
    ]',
    '{"mae": 310.5, "rmse": 427.2, "mape": 2.12, "r2": 0.871}',
    '{"lag_1": 0.42, "ev_sales": 0.25, "brine_supply": 0.18, "lag_2": 0.15}',
    now() - INTERVAL '11 hours'
)
ON CONFLICT (id) DO NOTHING;

-- Steel (HRC) — 12-month XGBoost point forecast (monthly, USD/tonne)
INSERT INTO forecasts (id, commodity, model_type, point_forecast, metrics, feature_importance, created_at)
VALUES (
    'f0000000-0000-0000-0000-000000000003',
    'Steel',
    'xgboost',
    '[
        {"date": "2026-01", "value": 715},
        {"date": "2026-02", "value": 728},
        {"date": "2026-03", "value": 742},
        {"date": "2026-04", "value": 735},
        {"date": "2026-05", "value": 750},
        {"date": "2026-06", "value": 768},
        {"date": "2026-07", "value": 755},
        {"date": "2026-08", "value": 762},
        {"date": "2026-09", "value": 780},
        {"date": "2026-10", "value": 772},
        {"date": "2026-11", "value": 785},
        {"date": "2026-12", "value": 798}
    ]',
    '{"mae": 11.8, "rmse": 16.2, "mape": 1.62, "r2": 0.908}',
    '{"lag_1": 0.35, "iron_ore": 0.22, "coking_coal": 0.20, "china_capacity": 0.23}',
    now() - INTERVAL '6 hours'
)
ON CONFLICT (id) DO NOTHING;
