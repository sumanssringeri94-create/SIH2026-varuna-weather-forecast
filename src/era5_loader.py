import cfgrib
import pandas as pd
import numpy as np

ERA5_FILE = r"data\raw\observations\9885cbd53ccb07271e5e0f53ca219b14.grib"

print("Reading ERA5 observations...")

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
# Extract main weather variables
# ---------------------------------------------------------

temperature = main_ds["t2m"]
pressure = main_ds["sp"]
u10 = main_ds["u10"]
v10 = main_ds["v10"]

# Calculate wind speed
wind_speed = np.sqrt(u10**2 + v10**2)

# ---------------------------------------------------------
# Convert main dataset to DataFrame
# ---------------------------------------------------------

df = main_ds[
    ["t2m", "sp", "u10", "v10"]
].to_dataframe().reset_index()

# Add calculated wind speed
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
# Convert units
# ---------------------------------------------------------

# Kelvin -> Celsius
df["temperature"] = df["t2m"] - 273.15

# Pascal -> hPa
df["pressure"] = df["sp"] / 100

# Remove unnecessary columns
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

# Calculate valid time
precip_df["valid_time"] = (
    precip_df["time"] + precip_df["step"]
)

# Keep only required columns
precip_df = precip_df[
    [
        "valid_time",
        "latitude",
        "longitude",
        "rainfall"
    ]
]

# ---------------------------------------------------------
# Match precipitation with observation time
# ---------------------------------------------------------

df = df.merge(
    precip_df,
    left_on=["time", "latitude", "longitude"],
    right_on=["valid_time", "latitude", "longitude"],
    how="left"
)

# Remove helper column
df = df.drop(columns=["valid_time"])

# Convert precipitation from metres to millimetres
df["rainfall"] = df["rainfall"] * 1000

# ---------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------

df = df.sort_values(
    ["time", "latitude", "longitude"]
).reset_index(drop=True)

print("\nERA5 observations processed successfully!")

print("\nNumber of records:")
print(len(df))

print("\nNumber of timestamps:")
print(df["time"].nunique())

print("\nTime range:")
print(df["time"].min(), "to", df["time"].max())

print("\nFirst 10 records:")
print(df.head(10))

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = "data/processed/era5_observations_real.csv"

df.to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")