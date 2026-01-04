# PM2.5 Air Quality Forecasting

**Long-Term PM2.5 Prediction for Beijing, New Delhi, and Kathmandu (2017-2030)**

[![Status](https://img.shields.io/badge)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-brightgreen)]()

---

##  Project Summary

This project forecasts PM2.5 air pollution levels using machine learning. We compared four models across three cities and generated predictions through 2030.

**Key Result**: **XGBoost won decisively**, achieving 25-47% better accuracy than deep learning alternatives.

### Quick Stats

| Metric | Value |
|--------|-------|
| **Cities** | Beijing, New Delhi, Kathmandu |
| **Historical Data** | 2017-2025 (8+ years) |
| **Test Period** | Jan 2024 - March 2025 |
| **Forecast Period** | 2026-2030 |
| **Models Compared** | XGBoost, LSTM, iTransformer, SARIMAX |
| **Best Model** | XGBoost (RMSE: 13.5-36.1 μg/m³) |

---

##  Repository Structure

```
DataAnalysis/
├── documentation.md              
├── README.md                     
├── requirements.txt
│
├── Data/                         # Prepared datasets
│   ├── Beijing_Ready.csv
│   ├── NewDelhi_Ready.csv
│   └── Kathmandu_Ready.csv
│
├── 01_Data/01_Raw/              # Original + future weather
│   └── Prediction_files/        # 2026-2030 scenarios
│
├── Results/                      # All outputs
│   ├── Model_Comparison_PyTorch.csv
│   ├── PyTorch_Dashboard.png
│   ├── Comprehensive_Analysis_PyTorch.png
│   ├── [City]_Annual_Forecasts_2026_2030.png
│   └── Forecast_2026_2030_LongTerm.png
│
└── scripts/                      # Python code
    ├── pm25_forecasting_pytorch.py
    ├── generate_future_forecasts.py
    ├── prepare_data_march2025.py
    ├── view_results_pytorch.py
    └── create_comprehensive_plots.py
```

---

##  Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Data preparation
python scripts/prepare_data_march2025.py

# Model training & validation (2024-2025)
python scripts/pm25_forecasting_pytorch.py

# Future forecasting (2026-2030)
python scripts/generate_future_forecasts.py

# View results
python scripts/view_results_pytorch.py
```

---

## 📊 Results at a Glance

### Model Performance (Test Set: 2024-2025)

| City | XGBoost | LSTM | iTransformer | SARIMAX |
|------|---------|------|--------------|---------|
| **Beijing** | **28.82** ⭐ | 44.91 | 40.91 | 149.76 |
| **New Delhi** | **36.05** ⭐ | 47.83 | 53.85 | 183.89 |
| **Kathmandu** | **13.53** ⭐ | 25.37 | 29.41 | 83.33 |

*RMSE values in μg/m³. Lower is better.*

### 2026-2030 Forecasts

| City | Predicted Mean PM2.5 | Peak Level | Trend |
|------|---------------------|------------|-------|
| Beijing | 108 μg/m³ | 219 μg/m³ | Stable |
| New Delhi | 168 μg/m³ | 349 μg/m³ | No improvement |
| Kathmandu | 115 μg/m³ | 164 μg/m³ | Declining |

---

##  Methodology

**Training**: 2017-2023 (7 years)  
**Testing**: 2024-March 2025 (15 months)  
**Features**: Weather (Temp, Wind, Humidity, Precip) + Temporal encoding + Lag features

### Models

1. **XGBoost**: 500 trees, 30 features (weather + lags + rolling stats)
2. **LSTM**: 3-layer PyTorch model, 30-day lookback
3. **iTransformer**: 2-layer transformer with attention, 30-day lookback
4. **SARIMAX**: Seasonal ARIMA with exogenous weather variables

---

##  Documentation

**For complete details, see [documentation.md](documentation.md)**

The full documentation includes:
- Detailed methodology
- Data preprocessing steps
- Model architectures
- Performance analysis
- Forecasting interpretation
- Limitations and caveats
- Future recommendations

---

## Key Visualizations

All plots are in the `Results/` folder:

- **PyTorch_Dashboard.png**: 3×3 grid showing performance + test data
- **Comprehensive_Analysis_PyTorch.png**: Full multi-panel comparison
- **[City]_Annual_Forecasts_2026_2030.png**: Year-by-year breakdowns
- **Forecast_2026_2030_LongTerm.png**: 5-year trend for all cities

---

## Tech Stack

- **Python 3.13**
- **PyTorch 2.9** (LSTM, iTransformer)
- **XGBoost 2.0+**
- **statsmodels** (SARIMAX)
- **pandas, numpy, matplotlib, seaborn**

---

## Key Findings

1. **XGBoost dominates** all cities by 25-47%
2. **Lag features matter most**: Yesterday's pollution is the best predictor
3. **Deep learning underperformed**: LSTM/iTransformer good but not better than XGBoost
4. **SARIMAX failed**: 5-7x worse errors, linear assumptions too rigid
5. **Seasonal patterns preserved**: Winter spikes and summer troughs correctly forecast

---

## Contributors

- Nilima Shrestha - [nillimaa373sths@gmail.com](mailto:nillimaa373sths@gmail.com)
- Aayush Acharya - [acharyaaayush2k4@gmail.com](mailto:acharyaaayush2k4@gmail.com)

---

## 📄 License

This project is for educational and research purposes.

---

## 🔗 Repository

**GitHub**: [nilima-sth/AQI-Prediction](https://github.com/nilima-sth/AQI-Prediction)

---

## Important Note

**March 2025 Cutoff**: Test data limited to March 31, 2025 due to Kathmandu data quality constraints. This ensures scientific integrity across all cities.

---

**Last Updated**: January 3, 2026  
**Recommended Model**: XGBoost with lag features


# Models Directory

Contains trained model weights, checkpoints, and configurations.

## Subdirectories:
- `weights/` - Final trained model files (.h5, .pkl, .json)
- `training_checkpoints/` - Intermediate checkpoints during training

## Models:
1. XGBoost - Gradient boosting for non-linear patterns
2. ARIMA/SARIMA - Statistical baseline with seasonality
3. Bi-LSTM - Recurrent neural network for sequences
4. iTransformer - Transformer architecture for multivariate time series