import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# File paths
# =========================================================

INPUT_FILE = "data/processed/forecast_comparison.csv"
OUTPUT_FILE = "data/processed/ml_predictions.csv"

print("Loading forecast comparison data...")

df = pd.read_csv(INPUT_FILE)

df["forecast_time"] = pd.to_datetime(df["forecast_time"])

print("\nDataset shape:")
print(df.shape)


# =========================================================
# Required columns
# =========================================================

required_columns = [
    "latitude",
    "longitude",
    "forecast_temperature",
    "forecast_pressure",
    "forecast_wind_speed",
    "forecast_rainfall_6h",
    "temperature",
    "pressure",
    "wind_speed",
    "rainfall",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )


# =========================================================
# Cleanup
# =========================================================

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

df = df.sort_values(
    "forecast_time"
).reset_index(drop=True)

print("\nRecords after cleanup:")
print(len(df))


# =========================================================
# Features
# =========================================================

features = [
    "latitude",
    "longitude",
    "forecast_temperature",
    "forecast_pressure",
    "forecast_wind_speed",
    "forecast_rainfall_6h",
]

X = df[features]


# =========================================================
# Targets
# =========================================================

targets = {
    "temperature": "temperature",
    "pressure": "pressure",
    "wind_speed": "wind_speed",
    "rainfall": "rainfall",
}


# =========================================================
# Time-based train/test split
#
# First 80% = training
# Last 20%  = testing
# =========================================================

unique_times = sorted(
    df["forecast_time"].unique()
)

split_index = int(
    len(unique_times) * 0.80
)

train_times = unique_times[:split_index]
test_times = unique_times[split_index:]

train_mask = df["forecast_time"].isin(train_times)
test_mask = df["forecast_time"].isin(test_times)

train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]

print("\nTraining period:")
print(
    pd.Timestamp(train_times[0]),
    "to",
    pd.Timestamp(train_times[-1])
)

print("\nTesting period:")
print(
    pd.Timestamp(test_times[0]),
    "to",
    pd.Timestamp(test_times[-1])
)

print("\nTraining records:")
print(len(train_indices))

print("\nTesting records:")
print(len(test_indices))


# =========================================================
# Train models
# =========================================================

models = {}
predictions = {}

for name, target_column in targets.items():

    print(
        f"\nTraining model for {name}..."
    )

    y = df[target_column]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    models[name] = model
    predictions[name] = prediction


# =========================================================
# Evaluate ML models
# =========================================================

print("\n")
print("=" * 60)
print("ML MODEL PERFORMANCE")
print("=" * 60)

for name, target_column in targets.items():

    y_test = df[
        target_column
    ].iloc[test_indices]

    prediction = predictions[name]

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    print(f"\n{name.upper()}")

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


# =========================================================
# Compare GFS vs ML
# =========================================================

print("\n")
print("=" * 60)
print("GFS BASELINE vs ML MODEL")
print("=" * 60)

baseline_columns = {
    "temperature": "forecast_temperature",
    "pressure": "forecast_pressure",
    "wind_speed": "forecast_wind_speed",
    "rainfall": "forecast_rainfall_6h",
}


for name, target_column in targets.items():

    y_test = df[
        target_column
    ].iloc[test_indices]

    baseline_prediction = df[
        baseline_columns[name]
    ].iloc[test_indices]

    ml_prediction = predictions[name]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction
    )

    ml_mae = mean_absolute_error(
        y_test,
        ml_prediction
    )

    improvement = (
        (baseline_mae - ml_mae)
        / baseline_mae
        * 100
        if baseline_mae != 0
        else 0
    )

    print(f"\n{name.upper()}")

    print(
        f"GFS MAE: {baseline_mae:.4f}"
    )

    print(
        f"ML  MAE: {ml_mae:.4f}"
    )

    print(
        f"Improvement: {improvement:.2f}%"
    )


# =========================================================
# Save predictions
# =========================================================

sample = df.iloc[test_indices].copy()

sample["ml_temperature"] = predictions[
    "temperature"
]

sample["ml_pressure"] = predictions[
    "pressure"
]

sample["ml_wind_speed"] = predictions[
    "wind_speed"
]

sample["ml_rainfall"] = predictions[
    "rainfall"
]


# =========================================================
# Save CSV
# =========================================================

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 60)
print("ML TRAINING COMPLETED")
print("=" * 60)

print("\nPredictions saved to:")
print(OUTPUT_FILE)