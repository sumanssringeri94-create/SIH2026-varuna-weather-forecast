import pandas as pd

JUNE_FILE = "data/processed/era5_observations_real.csv"
JULY_FILE = "data/processed/era5_observations_july2022.csv"

OUTPUT_FILE = "data/processed/era5_observations_june_july2022.csv"

print("Loading June ERA5...")
june = pd.read_csv(JUNE_FILE)

print("Loading July ERA5...")
july = pd.read_csv(JULY_FILE)

# Convert time
june["time"] = pd.to_datetime(june["time"])
july["time"] = pd.to_datetime(july["time"])

# Combine
df = pd.concat(
    [june, july],
    ignore_index=True
)

# Remove duplicate records if any
df = df.drop_duplicates(
    subset=["time", "latitude", "longitude"]
)

# Sort
df = df.sort_values(
    ["time", "latitude", "longitude"]
).reset_index(drop=True)

print("\n==============================================")
print("COMBINED ERA5 DATASET")
print("==============================================")

print("\nRecords:", len(df))
print("Timestamps:", df["time"].nunique())
print("Start:", df["time"].min())
print("End:", df["time"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRecords by month:")
print(df["time"].dt.month.value_counts().sort_index())

# Save
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)