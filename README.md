# 🌧️ VARUNA — Weather Forecast Intelligence & Forecast Bust Detection

**Varuna** is an ML-powered weather forecasting and forecast reliability analysis system designed to improve numerical weather prediction forecasts and identify locations where forecasts may deviate significantly from expected weather conditions.

The system combines **GFS forecast data**, **ERA5 observations**, feature engineering, machine learning, and a **Forecast Bust Detection** framework to generate corrected weather predictions and classify forecast reliability.

# Demo working prototype of VARUNA
https://drive.google.com/file/d/1k2SH05nmXf9UFjFA7HBrpyaUkqJdxmCx/view?usp=drive_link

---

## 🎯 Objectives

* Improve GFS weather forecasts using machine learning.
* Analyze forecast errors across multiple weather variables.
* Generate ML-corrected weather predictions.
* Detect potentially unreliable forecasts.
* Identify high-risk geographical locations.
* Determine the dominant weather factor contributing to forecast uncertainty.
* Provide visual analysis through an interactive Streamlit dashboard.

---

## 🧠 How Varuna Works

```text
GFS Forecast Data
       │
       ▼
Data Processing
       │
       ▼
Feature Engineering
       │
       ▼
ML Forecast Models
       │
       ├── Temperature
       ├── Pressure
       ├── Wind Speed
       └── Rainfall
       │
       ▼
ML-Corrected Forecast
       │
       ▼
Forecast Bust Detection
       │
       ├── Bust Score
       ├── Risk Level
       └── Dominant Factor
       │
       ▼
Varuna Dashboard
```

ERA5 observations are used as the reference dataset for model evaluation.

---

## 📊 Weather Variables

Varuna works with four major weather parameters:

| Variable    | Purpose                        |
| ----------- | ------------------------------ |
| Temperature | Correct temperature forecast   |
| Pressure    | Correct atmospheric pressure   |
| Wind Speed  | Correct wind-speed forecast    |
| Rainfall    | Correct precipitation forecast |

---

## ⚙️ Feature Engineering

The models use spatial, temporal, cyclic, and GFS forecast features.

```text
latitude
longitude
month
day
day_of_year
day_of_week
day_sin
day_cos
gfs_temperature
gfs_pressure
gfs_wind_speed
gfs_rainfall
```

### Feature Categories

**Spatial**

* Latitude
* Longitude

**Temporal**

* Month
* Day
* Day of year
* Day of week

**Cyclic**

* Day sine
* Day cosine

**Forecast**

* GFS temperature
* GFS pressure
* GFS wind speed
* GFS rainfall

---

# 🤖 Machine Learning

Varuna uses separate trained models for each weather variable.

```text
models/
├── temperature_model.pkl
├── pressure_model.pkl
├── wind_speed_model.pkl
└── rainfall_model.pkl
```

The models learn the relationship between GFS forecast features and the corresponding ERA5 observations.

---

# 📈 Model Performance

The improved models were evaluated on a chronological test period.

### Temperature

| Metric |    GFS |         ML |
| ------ | -----: | ---------: |
| MAE    | 1.0553 | **0.7247** |
| RMSE   | 1.3120 | **0.9539** |
| R²     | 0.5402 | **0.7570** |

**MAE Improvement: 31.32%**

### Pressure

| Metric |    GFS |         ML |
| ------ | -----: | ---------: |
| MAE    | 1.4403 | **0.6141** |
| RMSE   | 2.0112 | **0.8006** |
| R²     | 0.9958 | **0.9993** |

**MAE Improvement: 57.36%**

### Wind Speed

| Metric |        GFS |     ML |
| ------ | ---------: | -----: |
| MAE    | **0.9596** | 1.2953 |
| RMSE   | **1.1853** | 1.6598 |
| R²     | **0.5649** | 0.1468 |

**MAE Change: -34.99%**

The current ML wind-speed model does not outperform the GFS baseline and requires further improvement.

### Rainfall

| Metric |      GFS |         ML |
| ------ | -------: | ---------: |
| MAE    |   0.9920 | **0.3004** |
| RMSE   |   1.8110 | **0.4619** |
| R²     | -22.2079 |    -0.5098 |

**MAE Improvement: 69.72%**

---

# 🚨 Forecast Bust Detection

A key feature of Varuna is its **Forecast Bust Detection** system.

Instead of only generating a corrected forecast, Varuna measures how much the ML prediction differs from the original GFS forecast.

For every forecast location and timestamp, the system calculates:

```text
Temperature Correction
Pressure Correction
Wind Correction
Rainfall Correction
```

The magnitude of these corrections is converted into normalized risk values.

---

# 📊 Bust Score

The overall bust score combines the four weather-variable risks.

```text
Temperature : 30%
Pressure    : 20%
Wind        : 20%
Rainfall    : 30%
```

The final score is constrained between:

```text
0 and 1
```

A higher score indicates greater disagreement between the original GFS forecast and the ML-corrected forecast.

---

# ⚠️ Risk Classification

|    Bust Score | Classification |
| ------------: | -------------- |
|      `< 0.25` | NORMAL         |
| `0.25 – 0.49` | WATCH          |
| `0.50 – 0.74` | WARNING        |
|     `>= 0.75` | CRITICAL       |

Varuna also identifies the **dominant weather factor** based on the largest correction.

Possible factors:

```text
TEMPERATURE
PRESSURE
WIND
RAINFALL
```

---

# 🧪 Evaluation Dataset

The model was evaluated using a chronological train/test split.

### Training Period

```text
2022-06-01 06:00:00
to
2022-07-18 06:00:00
```

**Training records:** 1920

### Testing Period

```text
2022-07-19 06:00:00
to
2022-07-30 06:00:00
```

**Testing records:** 480

The chronological split ensures that the model is evaluated on a later period rather than randomly mixing training and testing observations.

---

# 📍 Forecast Risk Results

For the 480-record evaluation period:

| Risk Level | Count | Percentage |
| ---------- | ----: | ---------: |
| NORMAL     |   192 |     40.00% |
| WATCH      |   226 |     47.08% |
| WARNING    |    60 |     12.50% |
| CRITICAL   |     2 |      0.42% |

The highest-risk location identified in the evaluation was:

```text
Latitude: 15.0
Longitude: 74.0

Average Bust Score: 0.536371
Maximum Bust Score: 0.831064
Critical Forecasts: 1
Warning Forecasts: 5
```

---

# 📊 Generated Outputs

Varuna generates model evaluation, forecast-risk, and visualization files.

```text
data/
└── processed/
    └── evaluation/
        ├── model_performance.csv
        ├── critical_forecasts.csv
        ├── daily_risk.csv
        ├── location_risk.csv
        ├── gfs_vs_ml_mae.png
        ├── ml_improvement.png
        ├── risk_distribution.png
        ├── dominant_factors.png
        └── daily_bust_risk.png
```

Forecast comparison plots include:

```text
temperature_comparison.png
pressure_comparison.png
wind_speed_comparison.png
rainfall_comparison.png
```

---

# 🖥️ Varuna Dashboard

Varuna includes a **Streamlit dashboard** for visualizing the generated forecast and analysis.

The dashboard can be launched locally using:

```powershell
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

The dashboard provides access to:

* Weather forecast information
* ML predictions
* Forecast corrections
* Forecast bust risk
* Risk classifications
* Dominant weather factors
* Location-based risk information
* Model evaluation results

---

# 📁 Project Structure

```text
SIH2026-varuna-weather-forecast/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── processed/
│       ├── forecast_comparison.csv
│       ├── final_forecast.csv
│       ├── forecast_bust_results.csv
│       ├── ml_predictions_improved.csv
│       │
│       └── evaluation/
│           ├── model_performance.csv
│           ├── critical_forecasts.csv
│           ├── daily_risk.csv
│           ├── location_risk.csv
│           └── visualization files
│
├── models/
│   ├── temperature_model.pkl
│   ├── pressure_model.pkl
│   ├── wind_speed_model.pkl
│   └── rainfall_model.pkl
│
└── src/
    ├── data_loader.py
    ├── era5_loader.py
    ├── gfs_loader.py
    ├── features.py
    ├── train_baseline.py
    ├── train_improved.py
    ├── predict_weather.py
    ├── plot_results.py
    ├── forecast_bust_detector.py
    ├── final_evaluation.py
    └── other processing scripts
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/sumanssringeri94-create/SIH2026-varuna-weather-forecast.git
cd SIH2026-varuna-weather-forecast
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

After installing the dependencies:

```powershell
streamlit run app.py
```

---

# 🔄 ML Pipeline

### Train the improved models

```powershell
python src\train_improved.py
```

### Generate ML predictions

```powershell
python src\predict_weather.py
```

### Detect forecast busts

```powershell
python src\forecast_bust_detector.py
```

### Run final evaluation

```powershell
python src\final_evaluation.py
```

### Generate comparison plots

```powershell
python src\plot_results.py
```

---

# 🔍 Evaluation Metrics

Varuna uses:

### MAE — Mean Absolute Error

Measures the average absolute prediction error.

**Lower is better.**

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more heavily.

**Lower is better.**

### R² — Coefficient of Determination

Measures how well the model explains the variation in the observed data.

**Higher is generally better.**

---

# ⚠️ Limitations

Varuna is currently a prototype/research system.

Current limitations include:

* The evaluation uses a limited historical period.
* Model performance varies across weather variables.
* The current wind-speed model does not outperform GFS.
* Rainfall prediction remains challenging.
* Forecast Bust Detection measures disagreement between GFS and ML forecasts and should not be interpreted as a guaranteed prediction of actual forecast failure.
* The current dashboard is primarily designed for demonstration and analysis.
* The system has not been validated for operational meteorological deployment.

---

# 🔮 Future Scope

Future versions of Varuna can include:

* More years of weather data
* Real-time GFS forecast ingestion
* Higher-resolution weather data
* Additional atmospheric variables
* Improved wind-speed prediction
* Advanced rainfall prediction
* XGBoost / LightGBM models
* LSTM and other temporal models
* Spatiotemporal deep learning
* Interactive geographic risk maps
* Automated forecast-bust alerts
* Location-based warnings
* Real-time API integration
* Historical forecast reliability analysis

---

# 🏆 Key Highlights

### Machine Learning Forecast Correction

Uses ML to improve selected GFS forecast variables.

### Forecast Bust Detection

Identifies forecasts where the ML correction indicates potentially significant forecast uncertainty.

### Risk Classification

Converts forecast deviations into:

```text
NORMAL → WATCH → WARNING → CRITICAL
```

### Spatial Risk Analysis

Identifies geographical locations with higher forecast-bust risk.

### Visual Analytics

Provides graphs and a Streamlit dashboard for easier interpretation.

---

# 💡 Core Idea

Traditional weather forecasting provides a prediction.

**Varuna adds another layer of information:**

> **How confident should we be in that forecast?**

By combining numerical weather forecasts, observations, machine learning, and forecast deviation analysis, Varuna aims to provide a more informative view of forecast reliability.

---

# 🏫 Project Information

**Project Name:** Varuna

**Project:** Weather Forecast Intelligence & Forecast Bust Detection

**Domain:** Machine Learning / Weather Forecasting / Data Science

**Developed for:** Smart India Hackathon (SIH)

---

# ⚠️ Disclaimer

Varuna is an academic prototype developed for research, demonstration, and decision-support purposes.

The predictions and risk classifications generated by this system should not be considered a replacement for official meteorological forecasts, warnings, or emergency guidance.



Then refresh your GitHub repository—the README will automatically appear on the repository homepage.
