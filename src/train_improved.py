import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import create_weather_features


# =========================================================
# File paths
# =========================================================

INPUT_FILE = "data/processed/forecast_comparison.csv"
OUTPUT_FILE = "data/processed/ml_predictions_improved.csv"
MODEL_DIR = "models"


print("Loading forecast comparison data...")

df = pd.read_csv(INPUT_FILE)

df["forecast_time"] = pd.to_datetime(
    df["forecast_time"]
)

print("\nOriginal dataset shape:")
print(df.shape)


# =========================================================
# Create leakage-free features
# =========================================================

df, feature_columns = create_weather_features(df)

print("\nFeatures used for training:")
print(feature_columns)


# =========================================================
# Required target columns
# =========================================================

targets = {
    "temperature": "temperature",
    "pressure": "pressure",
    "wind_speed": "wind_speed",
    "rainfall": "rainfall",
}


# =========================================================
# Remove missing values
# =========================================================

required_columns = (
    feature_columns
    + list(targets.values())
)

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

df = df.sort_values(
    "forecast_time"
).reset_index(drop=True)

print("\nRecords after cleanup:")
print(len(df))


# =========================================================
# Feature matrix
# =========================================================

X = df[feature_columns]


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

train_mask = df["forecast_time"].isin(
    train_times
)

test_mask = df["forecast_time"].isin(
    test_times
)

train_indices = np.where(
    train_mask
)[0]

test_indices = np.where(
    test_mask
)[0]


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
# Create model directory
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# Train models
# =========================================================

models = {}
predictions = {}


for name, target_column in targets.items():

    print(
        f"\nTraining improved model for {name}..."
    )

    y = df[target_column]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
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

    # -----------------------------------------------------
    # Save trained model
    # -----------------------------------------------------

    model_file = os.path.join(
        MODEL_DIR,
        f"{name}_model.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"Saved model: {model_file}"
    )


# =========================================================
# Evaluate improved models
# =========================================================

print("\n")
print("=" * 65)
print("IMPROVED ML MODEL PERFORMANCE")
print("=" * 65)


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
# Compare GFS vs improved ML
# =========================================================

print("\n")
print("=" * 65)
print("GFS BASELINE vs IMPROVED ML")
print("=" * 65)


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
# Save test predictions
# =========================================================

sample = df.iloc[
    test_indices
].copy()


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
# Save predictions CSV
# =========================================================

sample.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Final summary
# =========================================================

print("\n")
print("=" * 65)
print("IMPROVED ML TRAINING COMPLETED")
print("=" * 65)

print("\nPredictions saved to:")
print(OUTPUT_FILE)

print("\nModels saved in:")
print(MODEL_DIR)

print("\nSaved models:")

for name in targets:
    print(
        f" - {os.path.join(MODEL_DIR, name + '_model.pkl')}"
    )