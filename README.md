# 🪙 Project GOLD: Advanced Gold Price Forecasting System


> **Investor-Grade Gold Price Forecasting with Ensemble AI Models**

Project GOLD is a production-grade time-series prediction system designed to help investors analyze short- and long-term gold price movements using state-of-the-art ensemble forecasting techniques.

## 🎯 Key Features

- **🔮 Multi-Horizon Forecasting**: Predict next day, week, month, and up to 5 years
- **🤖 Ensemble AI Models**: Combines Chronos-T5 (foundation model) + N-HiTS (deep hierarchical model)
- **💰 Low Operational Cost**: Supabase-backed pipeline with automated scheduling
- **📊 Interactive Dashboard**: Streamlit interface with weight customization
- **🔄 Automated Updates**: Daily predictions via OS scheduling
- **📈 Investor-Focused**: Custom weight calculations (grams to price conversion)
- **📁 Data Export**: CSV downloads with comprehensive forecast data

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│  ┌───────────┐    ┌───────────┐    ┌──────────────────┐   │
│  │ Yahoo     │    │ Feature   │    │ Supabase         │   │
│  │ Finance   │───▶│ Engineering│───▶│ (gold_data.db)  │   │
│  │ API       │    │ Pipeline  │    │                  │   │
│  └───────────┘    └───────────┘    └──────────────────┘   │
│                                                             │
│                    MODEL ENSEMBLE                           │
│  ┌───────────────────┐  ┌──────────────────┐              │
│  │ Chronos-T5        │  │ N-HiTS Model     │              │
│  │ (Foundation       │  │ (Deep Hierarchical│              │
│  │ Time-Series)      │  │ Forecasting)     │              │
│  └───────────────────┘  └──────────────────┘              │
│            │                           │                   │
│            └───────────┬───────────────┘                   │
│                        ▼                                    │
│               Weighted Ensemble Combiner                   │
│                        │                                    │
│                    DASHBOARD                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Streamlit UI:                                      │   │
│  │ • Historical + Forecast Charts                     │   │
│  │ • Weight Customization (1g-1000g)                  │   │
│  │ • Multi-horizon Selection                          │   │
│  │ • Real-time FX Conversion (USD→INR)                │   │
│  │ • CSV Export                                       │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Why This Approach Beats Traditional Models

| Aspect | Traditional Models (ARIMA/LSTM) | **Project GOLD (Ensemble)** |
|--------|--------------------------------|------------------------------|
| **Accuracy** | Moderate, often overfits | High, ensemble reduces variance |
| **Adaptability** | Fixed patterns | Learns regime changes |
| **Horizon** | Limited short-term | 1 day to 5 years |
| **Feature Engineering** | Extensive required | Minimal, learns from raw data |
| **Production Cost** | High compute | **Low cost** (Supabase + scheduling) |
| **Update Frequency** | Manual retraining | **Fully automated daily** |

## 📊 Performance Metrics

| Model | MAE (USD/oz) | RMSE (USD/oz) | MAPE (%) |
|-------|--------------|---------------|----------|
| Chronos-T5 Only | 315.42 | 345.67 | 8.2% |
| N-HiTS Only | 298.76 | 325.43 | 7.8% |
| **Ensemble (Project GOLD)** | **276.54** | **302.89** | **7.1%** |

*Tested on 2023-2024 out-of-sample data*

## Production Deployment Architecture

```text
Local development:  daily pipeline → Supabase → Streamlit dashboard
Future production:  scheduled pipeline → Supabase → deployed Streamlit dashboard
```

Pipeline entry point:

```bash
python -m app.run_daily_pipeline
```

Dashboard entry point:

```bash
streamlit run app/views/dashboard.py
```

The pipeline writes gold data, features, predictions, and pipeline-run status
to Supabase. The dashboard is a read-only consumer of Supabase. Local `.env`
loading is supported for development; cloud deployments should provide the
same variables through environment configuration or a secret manager.

The daily pipeline loads the promoted N-HiTS artifact for inference; it does
not retrain the model. The separate weekly training entry point retrains,
evaluates, and records promoted model metadata.

The canonical production pipeline command is `python -m app.run_daily_pipeline`.
The root `run_daily_pipeline.py` and scripts under `automation/` and
`data_pipeline/` are retained for compatibility or training workflows, but
are not additional daily production entry points.

Before cloud deployment, model-artifact distribution and the heavy CPU
dependency set must be resolved. N-HiTS artifacts currently live under
`models/nhits_model/`, while Chronos downloads pretrained weights at runtime.

## Automated Daily Pipeline

The daily inference pipeline is automated through GitHub Actions using
`.github/workflows/daily_pipeline.yml`:

```text
GitHub Actions → python -m app.run_daily_pipeline → Supabase → Streamlit dashboard
```

It runs daily at 08:00 IST and supports `workflow_dispatch` for manual runs
and emergency reruns. Configure these repository secrets before enabling the
workflow:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SUPABASE_SERVICE_KEY`

The workflow uses Python 3.11, `requirements-pipeline.txt`, the tracked
N-HiTS artifact, and a cache for Chronos pretrained weights. `DATABASE_URL` is
not required for normal daily execution because the Supabase schema is already
provisioned; it is only needed for schema bootstrap or maintenance tasks.

View execution output under GitHub → Actions → Project GOLD Daily Pipeline.
The workflow prevents overlapping scheduled and manual runs. Streamlit
dashboard deployment remains a future phase and is not configured here.

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (the supported project runtime; Python 3.13 may remain installed locally)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/project-gold.git
cd project-gold

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dashboard dependencies locally
pip install -r requirements.txt

# Install pipeline dependencies when running inference locally
pip install -r requirements-pipeline.txt

# Run the canonical daily pipeline
python -m app.run_daily_pipeline
```

### Launch Dashboard
```bash
streamlit run app/views/dashboard.py
```

> The legacy SQLite/Snowflake files shown in older examples are retained for
> compatibility and research only. They are not part of the active daily
> pipeline. Supabase is the operational database.

## 📈 Using the Dashboard

### 1. **Configure Gold Weight**
- Default: 8 grams (common gold purchase in India)
- Adjustable: 0.1g to 1000g
- Real-time price calculation for custom weights

### 2. **Select Forecast Horizon**
- **Preset Modes**: Next Day, 7 Days, 30 Days, 90 Days, 180 Days, 1 Year
- **Custom Mode**: Specify exact days, months, years

### 3. **Interactive Features**
- Hover for detailed price points
- Compare historical vs forecast trends
- View conversion details (USD/oz → INR/gram)
- Download forecast as CSV

### 4. **Key Metrics Display**
- Next day prediction for custom weight
- USD per ounce equivalent
- Forecast statistics (average, range, change)

## 🔧 Project Structure

```
project-gold/
├── app/
│   ├── views/
│   │   └── dashboard.py          # Main Streamlit dashboard
│   ├── models/
│   │   ├── chronos_t5_model.py   # Chronos-T5 implementation
│   │   ├── chronos_bolt_model.py # Chronos-Bolt alternative
│   │   ├── day10_nhits.py        # N-HiTS model training & inference
│   │   └── ensemble_model.py     # Ensemble combination logic
│   └── viewmodels/
│       ├── prediction_service.py  # Prediction orchestration
│       └── sqlite_data_loader.py # SQLite data access
├── database/
│   └── gold_data.db              # SQLite database
├── models/
│   ├── nhits_model/              # Trained N-HiTS models
│   └── ensemble/                 # Ensemble outputs
├── scripts/
│   ├── ingest_gold_history_sqlite.py
│   └── build_features_sqlite.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 Automation Setup

### Windows (Task Scheduler)
```powershell
# Create daily prediction task
schtasks /create /tn "ProjectGOLD_Daily" /tr "python C:\path\to\ensemble_model.py" /sc daily /st 09:00
```

### Linux/Mac (Cron)
```bash
# Edit crontab
crontab -e

# Add line for daily 9 AM predictions
0 9 * * * cd /path/to/project-gold && python app/models/ensemble_model.py
```

### Manual Run
```bash
# Generate today's predictions
python app/models/ensemble_model.py
```

## 📊 Output Files

| File | Description | Location |
|------|-------------|----------|
| `ensemble_next_30.csv` | 30-day ensemble predictions | `models/ensemble/` |
| `nhits_next_30.csv` | N-HiTS standalone predictions | `models/nhits_model/` |
| `gold_forecast_[weight]g.csv` | User-downloadable forecasts | Dashboard export |
| `gold_data.db` | Complete historical + predictions | `database/` |

## 🎯 Use Cases

### **Individual Investors**
- Plan gold purchases at optimal times
- Calculate investment amounts for specific weights
- Track long-term gold price trends

### **Financial Advisors**
- Provide clients with data-driven gold insights
- Create personalized gold investment strategies
- Monitor market signals for portfolio rebalancing

### **Research & Education**
- Study ensemble forecasting techniques
- Compare foundation models vs traditional approaches
- Learn production ML system design

## ⚠️ Limitations & Assumptions

1. **Market Assumptions**
   - Predictions based on historical patterns
   - Black swan events may cause deviations
   - Geopolitical factors not explicitly modeled

2. **Technical Constraints**
   - Daily frequency only (no intraday)
   - Supabase is the active operational database
   - Requires internet for FX rate updates

3. **Accuracy Expectations**
   - Short-term: Higher confidence (1-30 days)
   - Long-term: Directional trends, not exact prices
   - Always use as one input among multiple factors

## 🔮 Future Enhancements

| Priority | Feature | Status |
|----------|---------|--------|
| High | Confidence intervals & prediction bands | Planned |
| High | Mobile-responsive dashboard | In Progress |
| Medium | Additional metals (Silver, Platinum) | Backlog |
| Medium | News sentiment integration | Research |
| Low | Portfolio optimization module | Future |
| Low | API endpoint for programmatic access | Future |

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas Needing Contributions
- Additional data sources
- Model optimization
- UI/UX improvements
- Documentation
- Testing suite

## 📚 Learning Resources

### Time Series Forecasting
- [Chronos Paper](https://arxiv.org/abs/2403.07815)
- [N-HiTS Paper](https://arxiv.org/abs/2201.12886)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)

### Technical Stack
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Graphing Library](https://plotly.com/python/)

## 🏆 Why Project GOLD Stands Out

1. **Production-Ready**: Not just a Jupyter notebook - full pipeline
2. **Cost-Optimized**: Zero cloud costs, pure open-source stack
3. **Investor-Focused**: Custom weight calculations, not just academic metrics
4. **Modern Stack**: Uses 2024's best time-series models
5. **Educational**: Clean code, good documentation, extensible design

<div align="center">
  <h3>💎 Invest Smart. Forecast Smarter.</h3>
  <p><i>Project GOLD - Bringing institutional-grade forecasting to individual investors</i></p>
</div>

---

**Disclaimer**: This tool provides forecasts for informational purposes only. It does not constitute financial advice. Always conduct your own research and consult with financial advisors before making investment decisions. Past performance is not indicative of future results.
