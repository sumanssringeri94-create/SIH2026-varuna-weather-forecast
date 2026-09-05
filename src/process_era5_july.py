import cfgrib
import pandas as pd
import numpy as np

ERA5_FILE = r"data\raw\observations\8e3ab4d89889da2bbf2d653bc4233080.grib"
OUTPUT_FILE = "data/processed/era5_observations_july2022.csv"

print("Reading July 2022 ERA5 observations...")

datasets = cfgrib.open_datasets(
    ERA5_FILE,
    backend_kwargs={"indexpath": ""}
)

print("Number of datasets:", len(datasets))

# ---------------------------------------------------------
# Find required datasets
# ---------------------------------------------------------

main_ds = None
precip_ds = None

for ds in datasets:
    if "t2m" in ds.data_vars:
        main_ds = ds

    if "tp" in ds.data_vars:
        precip_ds = ds

if main_ds is None:
    raise ValueError("Temperature dataset not found!")

if precip_ds is None:
    raise ValueError("Precipitation dataset not found!")

# ---------------------------------------------------------
# Main weather variables
# ---------------------------------------------------------

temperature = main_ds["t2m"]
pressure = main_ds["sp"]
u10 = main_ds["u10"]
v10 = main_ds["v10"]

# Wind speed
wind_speed = np.sqrt(u10**2 + v10**2)

# ---------------------------------------------------------
# Main dataframe
# ---------------------------------------------------------

df = main_ds[
    ["t2m", "sp", "u10", "v10"]
].to_dataframe().reset_index()

wind_speed_df = wind_speed.to_dataframe(
    name="wind_speed"
).reset_index()

df = df.merge(
    wind_speed_df[
        ["time", "latitude", "longitude", "wind_speed"]
    ],
    on=["time", "latitude", "longitude"],
    how="left"
)

# ---------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------

# Kelvin -> Celsius
df["temperature"] = df["t2m"] - 273.15

# Pascal -> hPa
df["pressure"] = df["sp"] / 100

# ---------------------------------------------------------
# Keep required columns
# ---------------------------------------------------------

df = df[
    [
        "time",
        "latitude",
        "longitude",
        "temperature",
        "pressure",
        "wind_speed"
    ]
]

# ---------------------------------------------------------
# Process precipitation
# ---------------------------------------------------------

precip_df = precip_ds["tp"].to_dataframe(
    name="rainfall"
).reset_index()

precip_df["valid_time"] = (
    precip_df["time"] + precip_df["step"]
)

precip_df = precip_df[
    [
        "valid_time",
        "latitude",
        "longitude",
        "rainfall"
    ]
]

# ---------------------------------------------------------
# Match precipitation to observation time
# ---------------------------------------------------------

df = df.merge(
    precip_df,
    left_on=["time", "latitude", "longitude"],
    right_on=["valid_time", "latitude", "longitude"],
    how="left"
)

df = df.drop(columns=["valid_time"])

# metres -> millimetres
df["rainfall"] = df["rainfall"] * 1000

# ---------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------

df = df.sort_values(
    ["time", "latitude", "longitude"]
).reset_index(drop=True)

print("\nJuly ERA5 processed successfully!")

print("\nRecords:")
print(len(df))

print("\nTimestamps:")
print(df["time"].nunique())

print("\nTime range:")
print(df["time"].min(), "to", df["time"].max())

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)