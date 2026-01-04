# Data Analysis Summary

## Executive Summary

### Difference between Beijing.csv and Beijing_final_datas.csv?

**Answer:**
- **Beijing.csv** = Meteorological data ONLY (Temperature, Wind, Humidity, Precipitation)
- **Beijing_final_datas.csv** = **Complete dataset** with BOTH meteorological + 6 AQI pollutants

### Understanding:
**Using files that have both meteorological and AQI for machine learning** - they contain both features (weather) and targets (air quality).

---

## File Comparison (Only for Beijing for understanding)

| Metric | Beijing.csv | Beijing_final_datas.csv |
|--------|-------------|------------------------|
| **Purpose** | Meteorological source data | ML-ready: Features + Targets |
| **Rows** | 4,289 | 4,263 |
| **Columns** | 7 | 13 |
| **Size** | 169.73 KB | 247.33 KB |
| **Variables** | Date, YEAR, DOY, Temp, Wind, Humidity, Precip | **+ pm25, pm10, o3, no2, so2, co** |
| **Date Range** | 2014-01-01 to 2025-09-28 | 2014-01-01 to 2025-09-02 |
| **Missing Data** | 4 rows with -999 (Sept 25-28, 2025) | None |
| **Continuity** | Complete daily sequence | Complete daily sequence |

---

## Detailed Analysis Results

### Beijing (Ready for Modeling)

**Meteorological Variables:**
- Temperature: -8.8°C to 30.6°C
- Wind Speed: 0.7 to 6.0 m/s
- Humidity: 33% to 98%
- Precipitation: 0 to 80 mm/day

**AQI Pollutants:**
- **PM2.5:** 9-508 µg/m³ (Mean: 107, Median: 98)
- **PM10:** 5-895 µg/m³ (Mean: 66, Median: 54)
- **O3:** 1-170 ppb (Mean: 43, Median: 35)
- **NO2:** 1-84 ppb (Mean: 19, Median: 17)
- **SO2:** 1-67 ppb (Mean: 3, Median: 1)
- **CO:** 1-88 ppm (Mean: 7, Median: 5)

**Data Quality:** 
- 11+ years of continuous data
- No gaps in time series
- Only 4 rows with -999 at end (easily removed)

---

### New Delhi (Ready for Modeling)

**AQI Pollutants (Higher than Beijing):**
- **PM2.5:** 29-693 µg/m³ (Mean: **182**, Median: 163) - 70% worse than Beijing
- **PM10:** 8-941 µg/m³ (Mean: **153**, Median: 123) - 2.3x higher than Beijing
- **O3:** 1-271 ppb (Mean: 29, Median: 21)
- **NO2:** 1-129 ppb (Mean: 29, Median: 25)
- **SO2:** 1-82 ppb (Mean: 9, Median: 8)
- **CO:** 1-83 ppm (Mean: 12, Median: 10)

**Data Quality:**
- Same structure as Beijing
- 11+ years of continuous data
- Extends 1 day further than Beijing (to Sept 3, 2025)

---

### Kathmandu ( Critical Issues)

**Current State:**
- **INCOMPLETE** - Only 2 columns instead of 13
- Missing ALL meteorological variables
- Missing 5 AQI pollutants (only has PM2.5)
- Starts 3 years late (2017 vs 2014)
- 25 gaps in time series
- Contains forecast data (2026-2030)

**Available Data:**
- **PM2.5 (Column2):** 0-339 µg/m³ (Mean: 78, Median: 50)
- **Date Range:** 2017-03-03 to 2030-12-31
- **Real Data:** Only through ~January 2025
- **Forecast Data:** August 2025 - December 2030

**Gaps Identified:**
- Multiple 1-7 day gaps throughout 2021-2023
- **MAJOR GAP:** March 5 - August 7, 2025 (155 days!)
- Missing dates around festivals/holidays

---

## Cross-City Comparison

### Air Quality Rankings (PM2.5 Averages)

| Rank | City | PM2.5 Mean | PM2.5 Median | AQI Category |
|------|------|-----------|--------------|--------------|
| Best | **Kathmandu** | 78 µg/m³ | 50 µg/m³ | Moderate-Unhealthy |
| mild | **Beijing** | 107 µg/m³ | 98 µg/m³ | Unhealthy for Sensitive |
| Worst| **New Delhi** | 182 µg/m³ | 163 µg/m³ | Unhealthy |

**Note:** New Delhi's PM2.5 is **2.3x higher** than Kathmandu and **70% worse** than Beijing.

### Data Availability Matrix

| Feature | Beijing | Delhi | Kathmandu |
|---------|---------|-------|-----------|
| **Start Date** | 2014-01-01 | 2014-01-01 | **2017-03-03** |
| **End Date (AQI)** | 2024-12-31 | 2025-09-02 | 2025-01-29 |
| **Years of Data** | 11.0 | 11.7 | 7.9 |
| **PM2.5** | ✅ | ✅ | ✅ |
| **PM10** | ✅ | ✅ | ❌ |
| **O3** | ✅ | ✅ | ❌ |
| **NO2** | ✅ | ✅ | ❌ |
| **SO2** | ✅ | ✅ | ❌ |
| **CO** | ✅ | ✅ | ❌ |
| **Temperature** | ✅ | ✅ | ✅ (separate file) |
| **Wind** | ✅ | ✅ | ✅ (separate file) |
| **Humidity** | ✅ | ✅ | ✅ (separate file) |
| **Precipitation** | ✅ | ✅ | ✅ (separate file) |
| **Time Series** | Continuous | Continuous | **25 gaps** |

---

## 🎬 What To Do? Recommended Actions

### Immediate Actions (Phase 1) **COMPLETED**

-  **Analyze data structure** - DONE via `data_analysis_report.py`
-  **Create project directories** - DONE via `create_project_structure.py`
-  **Move files to proper locations** - DONE (all CSVs in `01_Data/01_Raw/`)
-  **Document findings** - DONE (this document + ACTION_PLAN.md)

### Next Steps (Phase 2) 🔄 **READY TO START**

#### Step 1: Fix Kathmandu Dataset
```python
# Create: fix_kathmandu.py
- Merge Kathmandu.csv (meteorological) + Kathmandu_final_datas.csv (AQI)
- Rename 'Column2' to 'pm25'
- Add NaN placeholders for missing pollutants
- Interpolate small gaps (<3 days)
- Truncate at 2025-01-29 (last real observation)
- Document all transformations in 05_Documentation/
```

#### Step 2: Data Standardization
```python
# Create: data_standardization.py
- Load all *_final_datas.csv files
- Convert -999 → NaN
- Remove rows with -999 values
- Standardize dates to YYYY-MM-DD
- Normalize column names:
  - pm25 → target_aqi
  - Temp → met_temp
  - Wind → met_wind
  - Humidity → met_humid
  - Precip → met_precip
- Output to 01_Data/02_Interim/
```

#### Step 3: Feature Engineering
```python
# Create: feature_generator.py
- Calculate lagged variables (T-1, T-7, T-30)
- Calculate rolling means (7-day, 30-day)
- Add temporal features (day_of_week, month, season)
- Create external events flags:
  - COVID period (2020-01-01 to 2022-12-31)
  - Festivals (Diwali, Lunar New Year, Dashain, Tihar)
  - Policy events (if data available)
- Output to 01_Data/03_Processed/
```

#### Step 4: Model Development
```python
# Priority order:
1. XGBoost (fastest, interpretable) - Start here
2. ARIMA/SARIMA (statistical baseline)
3. Bi-LSTM (if computational resources allow)
4. iTransformer (most complex, implement last)
```

---

## Technical options to decide on

### Decision 1: Kathmandu Handling Strategy

**Option A: Limited Analysis**
- Use available PM2.5 data only
- Document limitations clearly
- Merge with Kathmandu.csv for meteorological features
- Impute small gaps, truncate at 2025-01-29
- Cannot compare all 6 pollutants

**Option B: Exclude Completely**
- Focus on Beijing & Delhi (complete datasets)
- Faster development
- Lose geographic diversity
- Miss South Asian comparison

**Option C: Request New Data**
- Could get complete dataset
- May delay project significantly
- May not be available

### Decision 2: Training Window

**Option A: 2017-2024 (Aligned across cities)**
- Ensures fair comparison
- All cities have data
- Loses 3 years of Beijing/Delhi data

**Option B: 2014-2024 (Maximum for Beijing/Delhi)**
- More training data
- Cannot compare Kathmandu directly
- Better for city-specific models

### Decision:

**A: Chosen 2017-2023 for training and 2024-March,2025 for testing including all the other wind features**
- Ensures fair comparison
- All cities have data


