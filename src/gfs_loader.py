import os
import glob
import eccodes
import pandas as pd
import numpy as np


# =========================================================
# Paths
# =========================================================

INPUT_DIR = r"data\raw\ncmrwf_forecast"
OUTPUT_FILE = r"data\processed\gfs_forecast_real.csv"


# =========================================================
# Target region
# =========================================================

MIN_LAT = 11
MAX_LAT = 18

MIN_LON = 74
MAX_LON = 78


print("Reading GFS forecasts...")


# =========================================================
# Find all historical GFS files
# June + July 2022
# =========================================================

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "gfs_3_2022*_0000_006.grb2"
        )
    )
)

print("\nGFS files found:", len(files))

if len(files) == 0:
    raise RuntimeError(
        "No GFS files found!"
    )


# =========================================================
# Read GRIB message
# =========================================================

def read_message(handle):

    values = np.array(
        eccodes.codes_get_values(handle)
    )

    try:

        latitudes = np.array(
            eccodes.codes_get_array(
                handle,
                "latitudes"
            )
        )

        longitudes = np.array(
            eccodes.codes_get_array(
                handle,
                "longitudes"
            )
        )

    except Exception:

        # Build coordinates for regular latitude/longitude grid

        ni = eccodes.codes_get(
            handle,
            "Ni"
        )

        nj = eccodes.codes_get(
            handle,
            "Nj"
        )

        lat1 = eccodes.codes_get(
            handle,
            "latitudeOfFirstGridPointInDegrees"
        )

        lat2 = eccodes.codes_get(
            handle,
            "latitudeOfLastGridPointInDegrees"
        )

        lon1 = eccodes.codes_get(
            handle,
            "longitudeOfFirstGridPointInDegrees"
        )

        lon2 = eccodes.codes_get(
            handle,
            "longitudeOfLastGridPointInDegrees"
        )

        latitudes = np.linspace(
            lat1,
            lat2,
            nj
        )

        longitudes = np.linspace(
            lon1,
            lon2,
            ni
        )

        longitudes, latitudes = np.meshgrid(
            longitudes,
            latitudes
        )

        latitudes = latitudes.flatten()
        longitudes = longitudes.flatten()

    return (
        latitudes,
        longitudes,
        values
    )


# =========================================================
# Process one GFS file
# =========================================================

def process_file(filename):

    print(
        "\nProcessing:",
        os.path.basename(filename)
    )

    required = {
        "sp": None,
        "2t": None,
        "10u": None,
        "10v": None,
        "prate": None
    }

    handles = []

    with open(
        filename,
        "rb"
    ) as f:

        while True:

            handle = eccodes.codes_grib_new_from_file(
                f
            )

            if handle is None:
                break

            handles.append(handle)

            try:

                short_name = eccodes.codes_get(
                    handle,
                    "shortName"
                )

                if short_name in required:

                    # Keep only surface-level / required message

                    if required[short_name] is None:

                        required[short_name] = handle

                    else:

                        eccodes.codes_release(
                            handle
                        )

                else:

                    eccodes.codes_release(
                        handle
                    )

            except Exception:

                eccodes.codes_release(
                    handle
                )

    missing = [
        key
        for key, value in required.items()
        if value is None
    ]

    if missing:

        for handle in required.values():

            if handle is not None:

                eccodes.codes_release(
                    handle
                )

        raise RuntimeError(
            f"Missing variables in {filename}: {missing}"
        )


    # -----------------------------------------------------
    # Forecast time
    # -----------------------------------------------------

    first_handle = required["sp"]

    data_date = eccodes.codes_get(
        first_handle,
        "dataDate"
    )

    data_time = eccodes.codes_get(
        first_handle,
        "dataTime"
    )

    forecast_hour = eccodes.codes_get(
        first_handle,
        "forecastTime"
    )

    base_time = pd.to_datetime(
        str(data_date)
        + str(data_time).zfill(4),
        format="%Y%m%d%H%M"
    )

    forecast_time = (
        base_time
        + pd.Timedelta(
            hours=forecast_hour
        )
    )


    # -----------------------------------------------------
    # Extract variables
    # -----------------------------------------------------

    lat_sp, lon_sp, sp = read_message(
        required["sp"]
    )

    lat_t, lon_t, temperature = read_message(
        required["2t"]
    )

    lat_u, lon_u, u10 = read_message(
        required["10u"]
    )

    lat_v, lon_v, v10 = read_message(
        required["10v"]
    )

    lat_p, lon_p, prate = read_message(
        required["prate"]
    )


    # -----------------------------------------------------
    # Convert Kelvin to Celsius
    # -----------------------------------------------------

    temperature = (
        temperature - 273.15
    )


    # -----------------------------------------------------
    # Pressure Pa -> hPa
    # -----------------------------------------------------

    sp = sp / 100.0


    # -----------------------------------------------------
    # Wind speed
    # -----------------------------------------------------

    wind_speed = np.sqrt(
        u10 ** 2 + v10 ** 2
    )


    # -----------------------------------------------------
    # Create dataframe
    # -----------------------------------------------------

    df = pd.DataFrame({

        "forecast_time": forecast_time,

        "latitude": lat_sp,

        "longitude": lon_sp,

        "forecast_temperature": temperature,

        "forecast_pressure": sp,

        "forecast_wind_speed": wind_speed,

        "forecast_precip_rate": prate
    })


    # -----------------------------------------------------
    # Filter target region
    # -----------------------------------------------------

    df = df[
        (df["latitude"] >= MIN_LAT)
        & (df["latitude"] <= MAX_LAT)
        & (df["longitude"] >= MIN_LON)
        & (df["longitude"] <= MAX_LON)
    ]


    # -----------------------------------------------------
    # Cleanup handles
    # -----------------------------------------------------

    for handle in required.values():

        eccodes.codes_release(
            handle
        )


    print(
        "Grid points inside target region:",
        len(df)
    )

    return df


# =========================================================
# Process all files
# =========================================================

all_data = []

for filename in files:

    try:

        df = process_file(
            filename
        )

        all_data.append(
            df
        )

    except Exception as e:

        print(
            "ERROR processing:",
            filename
        )

        print(e)


# =========================================================
# Combine datasets
# =========================================================

if len(all_data) == 0:

    raise RuntimeError(
        "No GFS data was successfully processed!"
    )


gfs = pd.concat(
    all_data,
    ignore_index=True
)


# =========================================================
# Sort
# =========================================================

gfs = gfs.sort_values(
    [
        "forecast_time",
        "latitude",
        "longitude"
    ]
).reset_index(
    drop=True
)


# =========================================================
# Remove duplicates
# =========================================================

gfs = gfs.drop_duplicates(
    subset=[
        "forecast_time",
        "latitude",
        "longitude"
    ]
).reset_index(
    drop=True
)


# =========================================================
# Display results
# =========================================================

print("\n")
print("=" * 60)
print("GFS FORECAST DATASET")
print("=" * 60)

print(
    "\nNumber of files processed:",
    len(all_data)
)

print(
    "\nNumber of records:",
    len(gfs)
)

print(
    "\nNumber of forecast times:",
    gfs["forecast_time"].nunique()
)

print(
    "\nForecast time range:"
)

print(
    gfs["forecast_time"].min(),
    "to",
    gfs["forecast_time"].max()
)

print(
    "\nLatitude range:"
)

print(
    gfs["latitude"].min(),
    "to",
    gfs["latitude"].max()
)

print(
    "\nLongitude range:"
)

print(
    gfs["longitude"].min(),
    "to",
    gfs["longitude"].max()
)

print(
    "\nMissing values:"
)

print(
    gfs.isnull().sum()
)

print(
    "\nFirst 10 records:"
)

print(
    gfs.head(10).to_string(
        index=False
    )
)


# =========================================================
# Save
# =========================================================

gfs.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nGFS multi-day loading completed successfully!"
)