import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from error_calculation import (
    calculate_errors,
    identify_forecast_bust
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

actual_file = "data/processed/sample_weather.csv"
forecast_file = "data/processed/sample_forecast.csv"

data = calculate_errors(
    actual_file,
    forecast_file
)

data = identify_forecast_bust(data)


# --------------------------------------------------
# 2. Select REAL prediction features
# --------------------------------------------------

features = [
    "forecast_temperature",
    "forecast_rainfall",
    "forecast_wind_speed",
    "forecast_pressure"
]

X = data[features]
y = data["forecast_bust"]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Predict
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)