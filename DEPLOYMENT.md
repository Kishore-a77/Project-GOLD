# Streamlit Community Cloud Deployment Guide

## Overview

This dashboard is a **prediction consumer** that reads pre-computed forecasts from Supabase. It does **not** run any models, training, or pipelines locally.

## Architecture

```
GitHub Actions (daily cron)
    ↓
Daily prediction pipeline (run_daily_pipeline.py)
    ↓
Supabase (gold_prices, predictions, pipeline_runs, model_metadata)
    ↓
Streamlit Dashboard (reads only)
```

## Prerequisites

- A [Streamlit Community Cloud](https://streamlit.io/cloud) account
- A [Supabase](https://supabase.com/) project with the required tables created
- GitHub repository connected to Streamlit Community Cloud

## Deployment Steps

### 1. Prepare Your Repository

Ensure the following files are committed to your repository:

- `streamlit_app.py` — Streamlit entry point (root level)
- `requirements.txt` — Dashboard dependencies only (lightweight)
- `app/views/dashboard.py` — Dashboard implementation
- `database/supabase_schema.sql` — Supabase schema (run once via SQL Editor)

### 2. Create Supabase Schema

Before deploying, ensure your Supabase database has the required tables. Run the SQL in `database/supabase_schema.sql` via the Supabase SQL Editor, or let the daily pipeline bootstrap it automatically via `DATABASE_URL`.

Required tables:
- `gold_prices` — Historical gold OHLCV data
- `predictions` — Ensemble forecasts (idempotent upserts)
- `pipeline_runs` — Pipeline execution status
- `model_metadata` — Model version tracking

### 3. Deploy on Streamlit Community Cloud

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click **"New app"**
3. Select your GitHub repository and branch
4. Set **Main file path** to: `streamlit_app.py`
5. Set **Python version** to: `3.10` (or higher)
6. Click **"Deploy"**

### 4. Configure Secrets

In the Streamlit Community Cloud app settings, add the following secrets:

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL (e.g., `https://xxx.supabase.co`) | Yes |
| `SUPABASE_KEY` | Your Supabase anon/public key | Yes |

**How to get these values:**
1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. Copy the **Project URL** → `SUPABASE_URL`
4. Copy the **anon/public** key → `SUPABASE_KEY`

**Important:** Do NOT use the `service_role` key for the dashboard. The dashboard only needs read access to `gold_prices`, `predictions`, `pipeline_runs`, and `model_metadata` tables.

### 5. Verify Deployment

After deployment:

1. Open the app URL provided by Streamlit Cloud
2. The dashboard should load historical gold prices from Supabase
3. If the daily pipeline has run successfully, forecasts should appear automatically
4. Check the "🤖 Model & Pipeline Info" expander for pipeline status

## Required Secrets (Summary)

```toml
# .streamlit/secrets.toml (for local development)
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

For Streamlit Community Cloud, add these in the app settings under **Secrets**.

## Python Version

- **Required:** Python 3.10+
- **Recommended:** Python 3.10 (matches GitHub Actions runner)

## Main Application File

- **File:** `streamlit_app.py`
- **Alternative:** `app/views/dashboard.py` (if configuring custom main file in Streamlit Cloud)

## Requirements

The `requirements.txt` file contains only the dashboard dependencies:

```
streamlit>=1.24.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
supabase>=2.0
python-dotenv>=1.0.0
requests>=2.31.0
```

**Note:** Training/model dependencies (torch, darts, chronos, etc.) are in `requirements-pipeline.txt` and are **not** installed by Streamlit Cloud. This keeps the deployment lightweight and fast.

## Local Development

To run the dashboard locally:

```bash
# Install dashboard dependencies
pip install -r requirements.txt

# Set environment variables (or use .env file)
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"

# Run the dashboard
streamlit run streamlit_app.py
```

Or with the direct file:
```bash
streamlit run app/views/dashboard.py
```

## Troubleshooting

### "No gold price data available"
- Ensure the daily pipeline has run successfully at least once
- Check that `gold_prices` table exists in Supabase and has data
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correctly set in Streamlit secrets

### "No predictions found"
- Ensure the daily pipeline has generated predictions for the selected horizon
- Check that the `predictions` table exists and has data
- The dashboard uses `load_ensemble_forecast(horizon_key)` which filters by horizon

### "Error loading pipeline status"
- Ensure the `pipeline_runs` table exists in Supabase
- Check that the daily pipeline is recording run status correctly

### Import errors
- Ensure all dependencies in `requirements.txt` are installed
- Check that `supabase` package version is compatible (`>=2.0`)

## Security Notes

- **Never** commit `.env` files or expose Supabase credentials
- Use `st.secrets` for Streamlit Cloud (never hard-code credentials)
- The dashboard uses the Supabase **anon/public** key, which should have only read permissions on the required tables
- For production, consider enabling Row Level Security (RLS) in Supabase with appropriate policies

## Updating the Dashboard

To update the dashboard after deployment:

1. Make changes to `app/views/dashboard.py`
2. Commit and push to your repository
3. Streamlit Community Cloud will automatically redeploy

To update dependencies:
1. Modify `requirements.txt`
2. Commit and push
3. Streamlit Community Cloud will reinstall dependencies on next deploy

## Additional Notes

- The dashboard **does not** execute any model inference locally
- All predictions are pre-computed by the GitHub Actions daily pipeline
- The dashboard is a **pure consumer** of Supabase data
- Model retraining happens only in GitHub Actions (weekly workflow)
- Chronos-T5 model is downloaded from HuggingFace during pipeline execution (not in dashboard)
