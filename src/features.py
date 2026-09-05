import pandas as pd
import numpy as np


def create_weather_features(data):
    """
    Create ML features using only information available
    from the GFS forecast and forecast timestamp.
    """

    df = data.copy()

    # ---------------------------------------------------------
    # Ensure datetime
    # ---------------------------------------------------------

    df["forecast_time"] = pd.to_datetime(
        df["forecast_time"]
    )

    # ---------------------------------------------------------
    # Time-based features
    # ---------------------------------------------------------

    df["month"] = df["forecast_time"].dt.month
    df["day"] = df["forecast_time"].dt.day
    df["day_of_year"] = df["forecast_time"].dt.dayofyear
    df["day_of_week"] = df["forecast_time"].dt.dayofweek

    # Cyclic representation of day of year
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    # ---------------------------------------------------------
    # Location features
    # ---------------------------------------------------------

    df["latitude"] = df["latitude"]
    df["longitude"] = df["longitude"]

    # ---------------------------------------------------------
    # GFS forecast features
    # ---------------------------------------------------------

    df["gfs_temperature"] = (
        df["forecast_temperature"]
    )

    df["gfs_pressure"] = (
        df["forecast_pressure"]
    )

    df["gfs_wind_speed"] = (
        df["forecast_wind_speed"]
    )

    df["gfs_rainfall"] = (
        df["forecast_rainfall_6h"]
    )

    # ---------------------------------------------------------
    # Final ML feature columns
    # ---------------------------------------------------------

    feature_columns = [
        "latitude",
        "longitude",

        "month",
        "day",
        "day_of_year",
        "day_of_week",

        "day_sin",
        "day_cos",

        "gfs_temperature",
        "gfs_pressure",
        "gfs_wind_speed",
        "gfs_rainfall",
    ]

    return df, feature_columns


if __name__ == "__main__":

    INPUT_FILE = (
        "data/processed/forecast_comparison.csv"
    )

    print("Loading forecast comparison data...")

    data = pd.read_csv(INPUT_FILE)

    features, feature_columns = create_weather_features(
        data
    )

    print("\nFeature columns:")
    print(feature_columns)

    print("\nFeature dataset shape:")
    print(features.shape)

    print("\nFirst 10 feature records:")
    print(
        features[
            ["forecast_time"] + feature_columns
        ].head(10)
    )

    print("\nMissing values:")

    print(
        features[feature_columns]
        .isnull()
        .sum()
    )