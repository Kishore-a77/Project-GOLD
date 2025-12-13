🪙 Gold Price Forecasting System

Ensemble Forecasting using Chronos-T5 + N-HiTS | Snowflake + Streamlit

📌 Overview

This project is a production-grade gold price prediction system designed for investors.
Unlike traditional ML models (ARIMA, LSTM, XGBoost), it uses state-of-the-art time-series foundation and deep forecasting models, combined via an ensemble strategy to achieve higher accuracy and robustness across short-, medium-, and long-term horizons.

The system predicts:

Next day

Next week

Next month

Extended horizons (up to 5 years)

It is fully automated, cost-optimized, and deployed with a public dashboard.

🧠 Model Architecture (Why This Is Different)
🔹 Models Used
Model	Role
Chronos-T5 (Foundation Model)	Captures short-term market dynamics and micro-patterns
N-HiTS (Hierarchical Deep Model)	Learns long-term trends and seasonality
Weighted Ensemble	Combines strengths → lower variance, higher accuracy
🔹 Why Not Traditional ML?

Traditional models:

Require heavy feature engineering

Overfit noisy financial data

Break during volatility

This system:

Uses pretrained forecasting intelligence

Learns directly from raw time-series

Adapts better to regime changes

🏗️ System Architecture
                ┌────────────────────────┐
                │ External Data Sources   │
                │ (LBMA, Yahoo, FX API)   │
                └───────────┬────────────┘
                            │
                   (Daily Python Job)
                            │
                ┌───────────▼────────────┐
                │ Snowflake (X-Small WH)  │
                │ RAW → PROCESSED tables  │
                │ Auto-suspend enabled    │
                └───────────┬────────────┘
                            │
                   Snowpark / Python
                            │
        ┌───────────────┬───────────────┐
        │ Chronos-T5     │ N-HiTS         │
        │ Forecast       │ Forecast       │
        └───────────────┴───────────────┘
                            │
                     Weighted Ensemble
                            │
                ┌───────────▼────────────┐
                │ Streamlit Dashboard     │
                │ Cached + Public         │
                └────────────────────────┘

📊 Data Sources
Data	Source
Gold Price	LBMA, Yahoo Finance
USD Index	Yahoo Finance (DXY)
FX Rates	Frankfurt Exchange API
Macros	FRED (CPI, Bond Yield)

✔ Long historical coverage
✔ High reliability
✔ Industry-standard sources

⚙️ Data Engineering Design
✅ Zero-Waste Daily Task

Runs once per day

Rebuilds only last 30 rows

No full-table scans

No ML inside Snowflake

Warehouse auto-suspends in seconds

💰 Cost

Snowflake: ₹80–₹120/month

Streamlit Cloud: Free

Total infra: Near zero

📈 Performance Summary (Ensemble)
Metric	Value
MAE	~290
RMSE	~313
MAPE	~7.1%

📌 Ensemble consistently outperforms individual models.

🖥️ Dashboard Features

✔ Historical vs Forecast comparison
✔ Next day / week / month views
✔ Custom forecast horizon (up to 5 years)
✔ INR-converted values
✔ Downloadable CSV
✔ Cached Snowflake queries

▶️ How to Run Locally
# create environment
python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run dashboard
streamlit run app/views/dashboard.py

🚀 Deployment
Recommended:

Streamlit Cloud (Free)

Secrets stored securely:

SNOWFLAKE_ACCOUNT=xxx
SNOWFLAKE_USER=xxx
SNOWFLAKE_PASSWORD=xxx

⚠️ Limitations

Forecasts are probabilistic, not guarantees

Black-swan geopolitical events not explicitly modeled

Daily frequency (no intraday trading signals)

🔮 Future Improvements

Confidence intervals (quantile forecasts)

News-aware forecasting (LLM embeddings)

Regime detection (bull / bear gold cycles)

Portfolio optimization integration

Mobile-friendly UI

🧪 Testing Checklist

✔ Data alignment verified
✔ No Snowflake runaway compute
✔ Model inference stable
✔ Dashboard load < 3 seconds
✔ Cache working correctly

🎓 Project Value

This project demonstrates:

Modern time-series forecasting (2024-25 level)

Cloud-native data engineering

Cost-optimized production thinking

Investor-focused UX

📌 Author

Kishore A
Domain: Data Engineering + Time-Series Forecasting
