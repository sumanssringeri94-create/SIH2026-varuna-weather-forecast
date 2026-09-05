import pandas as pd
import numpy as np


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

GFS_FILE = "data/processed/gfs_forecast_real.csv"
ERA5_FILE = "data/processed/era5_observations_june_july2022.csv"
OUTPUT_FILE = "data/processed/forecast_comparison.csv"


print("Loading GFS forecast...")
gfs = pd.read_csv(GFS_FILE)

print("Loading ERA5 observations...")
era5 = pd.read_csv(ERA5_FILE)


# ---------------------------------------------------------
# Convert time columns
# ---------------------------------------------------------

gfs["forecast_time"] = pd.to_datetime(
    gfs["forecast_time"]
)

era5["time"] = pd.to_datetime(
    era5["time"]
)


# ---------------------------------------------------------
# Display information
# ---------------------------------------------------------

print("\nGFS shape:")
print(gfs.shape)

print("\nERA5 shape:")
print(era5.shape)

print("\nGFS forecast time:")
print(gfs["forecast_time"].unique())

print("\nERA5 time range:")
print(
    era5["time"].min(),
    "to",
    era5["time"].max()
)


# ---------------------------------------------------------
# Match forecast with ERA5
# ---------------------------------------------------------

print("\nMatching GFS forecast with ERA5 observations...")


comparison = pd.merge(
    gfs,
    era5,
    left_on=[
        "forecast_time",
        "latitude",
        "longitude"
    ],
    right_on=[
        "time",
        "latitude",
        "longitude"
    ],
    how="inner"
)


# ---------------------------------------------------------
# Remove duplicate time column
# ---------------------------------------------------------

comparison = comparison.drop(
    columns=["time"]
)


# ---------------------------------------------------------
# Convert GFS precipitation rate
#
# GFS prate is kg/m²/s.
# 1 kg/m² = 1 mm water.
#
# For a 6-hour forecast:
#
# rainfall(mm) =
# precipitation_rate × 21600 seconds
# ---------------------------------------------------------

comparison["forecast_rainfall_6h"] = (
    comparison["forecast_precip_rate"] * 21600
)


# ---------------------------------------------------------
# Calculate errors
# ---------------------------------------------------------

comparison["temperature_error"] = (
    comparison["forecast_temperature"]
    - comparison["temperature"]
)

comparison["pressure_error"] = (
    comparison["forecast_pressure"]
    - comparison["pressure"]
)

comparison["wind_error"] = (
    comparison["forecast_wind_speed"]
    - comparison["wind_speed"]
)

comparison["rainfall_error"] = (
    comparison["forecast_rainfall_6h"]
    - comparison["rainfall"]
)


# ---------------------------------------------------------
# Absolute errors
# ---------------------------------------------------------

comparison["absolute_temperature_error"] = (
    comparison["temperature_error"].abs()
)

comparison["absolute_pressure_error"] = (
    comparison["pressure_error"].abs()
)

comparison["absolute_wind_error"] = (
    comparison["wind_error"].abs()
)

comparison["absolute_rainfall_error"] = (
    comparison["rainfall_error"].abs()
)


# ---------------------------------------------------------
# Check matched records
# ---------------------------------------------------------

print("\nNumber of matched records:")
print(len(comparison))


if len(comparison) == 0:
    raise RuntimeError(
        "No GFS and ERA5 records matched. "
        "Check time and coordinates."
    )


# ---------------------------------------------------------
# Check missing values
# ---------------------------------------------------------

print("\nMissing values:")
print(comparison.isna().sum())


# ---------------------------------------------------------
# Calculate MAE
# ---------------------------------------------------------

temperature_mae = (
    comparison["absolute_temperature_error"].mean()
)

pressure_mae = (
    comparison["absolute_pressure_error"].mean()
)

wind_mae = (
    comparison["absolute_wind_error"].mean()
)

rainfall_mae = (
    comparison["absolute_rainfall_error"].mean()
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\nForecast comparison completed successfully!")

print("\nMean absolute errors:")

print(
    f"Temperature: {temperature_mae:.4f} °C"
)

print(
    f"Pressure: {pressure_mae:.4f} hPa"
)

print(
    f"Wind: {wind_mae:.4f} m/s"
)

print(
    f"Rainfall: {rainfall_mae:.4f} mm"
)


# ---------------------------------------------------------
# Display rainfall comparison
# ---------------------------------------------------------

print("\nRainfall comparison:")

print(
    comparison[
        [
            "latitude",
            "longitude",
            "forecast_precip_rate",
            "forecast_rainfall_6h",
            "rainfall",
            "absolute_rainfall_error"
        ]
    ].head(10).to_string(index=False)
)


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved to:")
print(OUTPUT_FILE)