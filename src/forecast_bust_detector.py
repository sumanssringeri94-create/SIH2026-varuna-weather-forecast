import pandas as pd
import numpy as np


# =========================================================
# File paths
# =========================================================

INPUT_FILE = "data/processed/final_forecast.csv"
OUTPUT_FILE = "data/processed/forecast_bust_results.csv"


print("=" * 65)
print("VALIDATED FORECAST BUST DETECTION")
print("=" * 65)


# =========================================================
# Load final forecast
# =========================================================

print("\nLoading final forecast...")

df = pd.read_csv(INPUT_FILE)

df["forecast_time"] = pd.to_datetime(
    df["forecast_time"]
)

print("Records:", len(df))


# =========================================================
# Validate required columns
# =========================================================

required_columns = [
    "forecast_time",
    "latitude",
    "longitude",

    "forecast_temperature",
    "forecast_pressure",
    "forecast_wind_speed",
    "forecast_rainfall_6h",

    "ml_temperature",
    "ml_pressure",
    "ml_wind_speed",
    "ml_rainfall",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# =========================================================
# Calculate ML correction
#
# Correction = ML corrected forecast - original GFS
# =========================================================

df["temperature_correction"] = (
    df["ml_temperature"]
    - df["forecast_temperature"]
)

df["pressure_correction"] = (
    df["ml_pressure"]
    - df["forecast_pressure"]
)

df["wind_correction"] = (
    df["ml_wind_speed"]
    - df["forecast_wind_speed"]
)

df["rainfall_correction"] = (
    df["ml_rainfall"]
    - df["forecast_rainfall_6h"]
)


# =========================================================
# Absolute corrections
# =========================================================

df["abs_temperature_correction"] = (
    df["temperature_correction"].abs()
)

df["abs_pressure_correction"] = (
    df["pressure_correction"].abs()
)

df["abs_wind_correction"] = (
    df["wind_correction"].abs()
)

df["abs_rainfall_correction"] = (
    df["rainfall_correction"].abs()
)


# =========================================================
# Correction severity
#
# These thresholds represent a meaningful difference
# between the original GFS forecast and ML-corrected
# forecast.
# =========================================================

temperature_score = (
    df["abs_temperature_correction"] / 2.0
)

pressure_score = (
    df["abs_pressure_correction"] / 3.0
)

wind_score = (
    df["abs_wind_correction"] / 2.5
)

rainfall_score = (
    df["abs_rainfall_correction"] / 5.0
)


# =========================================================
# Clip individual scores
#
# Prevent one extreme variable from dominating the
# complete risk score.
# =========================================================

temperature_score = temperature_score.clip(
    0,
    1
)

pressure_score = pressure_score.clip(
    0,
    1
)

wind_score = wind_score.clip(
    0,
    1
)

rainfall_score = rainfall_score.clip(
    0,
    1
)


# =========================================================
# Validated weighting
#
# Based on the observed ML validation:
#
# Temperature → ML improved
# Pressure    → ML improved strongly
# Wind        → ML did not improve
# Rainfall    → ML improved strongly
#
# Therefore wind receives lower weight.
# =========================================================

TEMPERATURE_WEIGHT = 0.30
PRESSURE_WEIGHT = 0.25
WIND_WEIGHT = 0.10
RAINFALL_WEIGHT = 0.35


# =========================================================
# Calculate weighted bust score
# =========================================================

df["bust_score"] = (

    TEMPERATURE_WEIGHT
    * temperature_score

    + PRESSURE_WEIGHT
    * pressure_score

    + WIND_WEIGHT
    * wind_score

    + RAINFALL_WEIGHT
    * rainfall_score
)


# =========================================================
# Ensure score is between 0 and 1
# =========================================================

df["bust_score"] = (
    df["bust_score"]
    .clip(0, 1)
)


# =========================================================
# Bust classification
# =========================================================

def classify_bust(score):

    if score < 0.25:

        return "NORMAL"

    elif score < 0.50:

        return "WATCH"

    elif score < 0.75:

        return "WARNING"

    else:

        return "CRITICAL"


df["forecast_bust"] = (
    df["bust_score"]
    .apply(classify_bust)
)


# =========================================================
# Determine dominant weather factor
#
# Use normalized severity rather than raw corrections.
# This makes the comparison fair between variables with
# different units.
# =========================================================

normalized_scores = {

    "TEMPERATURE": temperature_score,

    "PRESSURE": pressure_score,

    "WIND": wind_score,

    "RAINFALL": rainfall_score,
}


def get_dominant_factor(index):

    scores = {

        "TEMPERATURE":
            normalized_scores["TEMPERATURE"].iloc[index],

        "PRESSURE":
            normalized_scores["PRESSURE"].iloc[index],

        "WIND":
            normalized_scores["WIND"].iloc[index],

        "RAINFALL":
            normalized_scores["RAINFALL"].iloc[index],
    }

    return max(
        scores,
        key=scores.get
    )


df["dominant_factor"] = [

    get_dominant_factor(i)

    for i in range(len(df))
]


# =========================================================
# Add individual normalized scores to output
#
# These are useful for dashboard visualization and
# explanation of why a forecast received its risk score.
# =========================================================

df["temperature_risk"] = temperature_score

df["pressure_risk"] = pressure_score

df["wind_risk"] = wind_score

df["rainfall_risk"] = rainfall_score


# =========================================================
# Summary
# =========================================================

print("\n")
print("=" * 65)
print("VALIDATED FORECAST BUST SUMMARY")
print("=" * 65)


# ---------------------------------------------------------
# Classification counts
# ---------------------------------------------------------

print("\nBust classification:")

classification_counts = (
    df["forecast_bust"]
    .value_counts()
)

print(
    classification_counts
)


# ---------------------------------------------------------
# Dominant factors
# ---------------------------------------------------------

print("\nDominant factors:")

factor_counts = (
    df["dominant_factor"]
    .value_counts()
)

print(
    factor_counts
)


# ---------------------------------------------------------
# Score statistics
# ---------------------------------------------------------

print("\nBust score statistics:")

print(
    df["bust_score"]
    .describe()
)


# =========================================================
# Highest-risk forecasts
# =========================================================

print("\n")
print("=" * 65)
print("TOP 10 HIGHEST-RISK FORECASTS")
print("=" * 65)


top_risk = (

    df.sort_values(
        "bust_score",
        ascending=False
    )

    [[

        "forecast_time",
        "latitude",
        "longitude",

        "bust_score",
        "forecast_bust",
        "dominant_factor",

        "temperature_correction",
        "pressure_correction",
        "wind_correction",
        "rainfall_correction",

        "temperature_risk",
        "pressure_risk",
        "wind_risk",
        "rainfall_risk",

    ]]

    .head(10)
)


print(
    top_risk.to_string(
        index=False
    )
)


# =========================================================
# Highest-risk CRITICAL forecasts
# =========================================================

critical = (

    df[
        df["forecast_bust"]
        == "CRITICAL"
    ]

    .sort_values(
        "bust_score",
        ascending=False
    )

)


print("\n")
print("=" * 65)
print("CRITICAL FORECAST COUNT")
print("=" * 65)

print(
    len(critical)
)


# =========================================================
# Forecast period
# =========================================================

print("\nForecast period:")

print(
    df["forecast_time"].min(),
    "to",
    df["forecast_time"].max()
)


# =========================================================
# Check missing values
# =========================================================

print("\nMissing values:")

print(
    df.isna().sum()
)


# =========================================================
# Save results
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Final message
# =========================================================

print("\n")
print("=" * 65)
print("VALIDATED FORECAST BUST DETECTION COMPLETED")
print("=" * 65)

print("\nSaved to:")

print(
    OUTPUT_FILE
)

print("\nOutput records:")

print(
    len(df)
)

print("\nRisk levels:")

print(
    "NORMAL   : score < 0.25"
)

print(
    "WATCH    : 0.25 - 0.49"
)

print(
    "WARNING  : 0.50 - 0.74"
)

print(
    "CRITICAL : >= 0.75"
)