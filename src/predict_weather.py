import pandas as pd
import joblib
import os

from features import create_weather_features


# =========================================================
# File paths
# =========================================================

INPUT_FILE = "data/processed/forecast_comparison.csv"

OUTPUT_FILE = "data/processed/final_forecast.csv"

MODEL_DIR = "models"


print("=" * 65)
print("WEATHER FORECAST PREDICTION")
print("=" * 65)


# =========================================================
# Load data
# =========================================================

print("\nLoading GFS forecast data...")

df = pd.read_csv(INPUT_FILE)

df["forecast_time"] = pd.to_datetime(
    df["forecast_time"]
)

print("Records:", len(df))


# =========================================================
# Create features
# =========================================================

print("\nCreating prediction features...")

df, feature_columns = create_weather_features(df)

X = df[feature_columns]


print("Features used:")
print(feature_columns)


# =========================================================
# Load trained models
# =========================================================

print("\nLoading trained models...")

models = {}

for name in [
    "temperature",
    "pressure",
    "wind_speed",
    "rainfall"
]:

    model_file = os.path.join(
        MODEL_DIR,
        f"{name}_model.pkl"
    )

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    models[name] = joblib.load(
        model_file
    )

    print(
        f"Loaded: {model_file}"
    )


# =========================================================
# Generate predictions
# =========================================================

print("\nGenerating ML predictions...")

df["ml_temperature"] = models[
    "temperature"
].predict(X)

df["ml_pressure"] = models[
    "pressure"
].predict(X)

df["ml_wind_speed"] = models[
    "wind_speed"
].predict(X)

df["ml_rainfall"] = models[
    "rainfall"
].predict(X)


# =========================================================
# Prevent negative rainfall
# =========================================================

df["ml_rainfall"] = df[
    "ml_rainfall"
].clip(lower=0)


# =========================================================
# Select final output columns
# =========================================================

output_columns = [
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


result = df[
    output_columns
].copy()


# =========================================================
# Sort output
# =========================================================

result = result.sort_values(
    [
        "forecast_time",
        "latitude",
        "longitude"
    ]
).reset_index(drop=True)


# =========================================================
# Save final forecast
# =========================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Display summary
# =========================================================

print("\n")
print("=" * 65)
print("FINAL FORECAST GENERATED")
print("=" * 65)

print("\nRecords:", len(result))

print("\nForecast period:")
print(
    result["forecast_time"].min(),
    "to",
    result["forecast_time"].max()
)

print("\nFirst 10 predictions:")

print(
    result.head(10).to_string(
        index=False
    )
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nPrediction completed successfully!")