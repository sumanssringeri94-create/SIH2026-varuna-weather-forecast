import os
import glob
import pandas as pd
import numpy as np
import eccodes


# =========================================================
# VARUNA - GFS FORECAST LOADER
# Day 1 to Day 10 capable
# =========================================================


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_DIR = r"data\raw\ncmrwf_forecast"

OUTPUT_FILE = (
    r"data\processed\gfs_forecast_real.csv"
)


# =========================================================
# SUPPORTED FORECAST RANGE
# =========================================================

MAX_FORECAST_HOUR = 240


# =========================================================
# VARIABLES
# =========================================================

VARIABLES = {

    "temperature": [
        "2t",
        "TMP"
    ],

    "pressure": [
        "sp",
        "PRES"
    ],

    "wind_u": [
        "10u",
        "UGRD"
    ],

    "wind_v": [
        "10v",
        "VGRD"
    ],

    "precipitation": [
        "prate",
        "PRATE"
    ]
}


# =========================================================
# HEADER
# =========================================================

print("=" * 70)
print("VARUNA - GFS FORECAST LOADER")
print("DAY 1 - DAY 10 FORECAST SUPPORT")
print("=" * 70)


# =========================================================
# FIND GRIB FILES
# =========================================================

print("\nSearching for GFS GRIB files...")

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*.grb2"
        )
    )
)

print("Files found:", len(files))


if len(files) == 0:

    raise FileNotFoundError(
        "\nNo GFS .grb2 files found in:\n"
        + INPUT_DIR
        + "\n\nPlease download the required forecast files first."
    )


# =========================================================
# FORECAST DAY FUNCTION
# =========================================================

def get_forecast_day(forecast_hour):

    """
    Convert forecast lead hour to SIH forecast day.

    001-024  -> Day 1
    025-048  -> Day 2
    049-072  -> Day 3
    073-096  -> Day 4
    097-120  -> Day 5
    121-144  -> Day 6
    145-168  -> Day 7
    169-192  -> Day 8
    193-216  -> Day 9
    217-240  -> Day 10
    """

    forecast_hour = int(
        forecast_hour
    )

    if forecast_hour < 1:

        return 0

    forecast_day = int(
        np.ceil(
            forecast_hour / 24
        )
    )

    return forecast_day


# =========================================================
# FORECAST DAY LABEL
# =========================================================

def get_forecast_day_label(
    forecast_day
):

    if forecast_day <= 0:

        return "UNKNOWN"

    if forecast_day > 10:

        return "BEYOND_DAY_10"

    return (
        f"DAY_{forecast_day}"
    )


# =========================================================
# READ FIRST GRIB MESSAGE
# =========================================================

def read_first_message_metadata(
    filename
):

    handle = None

    try:

        with open(
            filename,
            "rb"
        ) as f:

            handle = (
                eccodes.codes_grib_new_from_file(
                    f
                )
            )

            if handle is None:

                return None


            # -------------------------------------------------
            # Initialization date
            # -------------------------------------------------

            data_date = eccodes.codes_get(
                handle,
                "dataDate"
            )


            # -------------------------------------------------
            # Initialization time
            # -------------------------------------------------

            data_time = eccodes.codes_get(
                handle,
                "dataTime"
            )

            data_time = int(
                data_time
            )


            hour = data_time // 100
            minute = data_time % 100


            # -------------------------------------------------
            # Base time
            # -------------------------------------------------

            base_time = (
                pd.Timestamp(
                    str(data_date)
                )
                + pd.Timedelta(
                    hours=hour,
                    minutes=minute
                )
            )


            # -------------------------------------------------
            # Forecast hour
            # -------------------------------------------------

            forecast_hour = eccodes.codes_get(
                handle,
                "forecastTime"
            )

            forecast_hour = int(
                forecast_hour
            )


            # -------------------------------------------------
            # Forecast time
            # -------------------------------------------------

            forecast_time = (
                base_time
                + pd.Timedelta(
                    hours=forecast_hour
                )
            )


            # -------------------------------------------------
            # Forecast day
            # -------------------------------------------------

            forecast_day = (
                get_forecast_day(
                    forecast_hour
                )
            )


            return {

                "forecast_initialization":
                    base_time,

                "forecast_hour":
                    forecast_hour,

                "forecast_day":
                    forecast_day,

                "forecast_day_label":
                    get_forecast_day_label(
                        forecast_day
                    ),

                "forecast_time":
                    forecast_time
            }


    except Exception as e:

        print(
            "Metadata error:",
            e
        )

        return None


    finally:

        if handle is not None:

            try:

                eccodes.codes_release(
                    handle
                )

            except Exception:

                pass


# =========================================================
# FIND VARIABLE MESSAGE
# =========================================================

def find_variable_message(
    filename,
    variable_names
):

    try:

        with open(
            filename,
            "rb"
        ) as f:

            while True:

                handle = None

                try:

                    handle = (
                        eccodes.codes_grib_new_from_file(
                            f
                        )
                    )

                except Exception:

                    break


                if handle is None:

                    break


                try:

                    short_name = (
                        str(
                            eccodes.codes_get(
                                handle,
                                "shortName"
                            )
                        )
                    )

                    name = (
                        str(
                            eccodes.codes_get(
                                handle,
                                "name"
                            )
                        )
                    )


                    for variable in variable_names:

                        if (
                            short_name.lower()
                            == variable.lower()
                            or
                            name.lower()
                            == variable.lower()
                        ):

                            return handle


                except Exception:

                    pass


                try:

                    eccodes.codes_release(
                        handle
                    )

                except Exception:

                    pass


    except Exception as e:

        print(
            "Variable search error:",
            e
        )


    return None


# =========================================================
# READ GRIB VARIABLE
# =========================================================

def read_grib_variable(
    filename,
    variable_names
):

    handle = find_variable_message(
        filename,
        variable_names
    )


    if handle is None:

        return None


    try:

        latitudes = (
            eccodes.codes_get_array(
                handle,
                "latitudes"
            )
        )

        longitudes = (
            eccodes.codes_get_array(
                handle,
                "longitudes"
            )
        )

        values = (
            eccodes.codes_get_array(
                handle,
                "values"
            )
        )


        return {

            "latitude":
                latitudes,

            "longitude":
                longitudes,

            "values":
                values
        }


    except Exception as e:

        print(
            "Reading variable failed:",
            e
        )

        return None


    finally:

        try:

            eccodes.codes_release(
                handle
            )

        except Exception:

            pass


# =========================================================
# PROCESS ONE GFS FILE
# =========================================================

def process_gfs_file(
    filename
):

    print(
        "\nProcessing:",
        os.path.basename(
            filename
        )
    )


    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = (
        read_first_message_metadata(
            filename
        )
    )


    if metadata is None:

        print(
            "Could not read metadata."
        )

        return None


    forecast_hour = (
        metadata[
            "forecast_hour"
        ]
    )


    forecast_day = (
        metadata[
            "forecast_day"
        ]
    )


    # -----------------------------------------------------
    # Ignore files beyond Day 10
    # -----------------------------------------------------

    if (
        forecast_hour < 1
        or
        forecast_hour > MAX_FORECAST_HOUR
    ):

        print(
            "Skipping forecast hour:",
            forecast_hour
        )

        return None


    print(
        "Initialization:",
        metadata[
            "forecast_initialization"
        ]
    )

    print(
        "Forecast hour:",
        forecast_hour
    )

    print(
        "Forecast day:",
        forecast_day
    )

    print(
        "Forecast label:",
        metadata[
            "forecast_day_label"
        ]
    )

    print(
        "Forecast time:",
        metadata[
            "forecast_time"
        ]
    )


    # -----------------------------------------------------
    # Read variables
    # -----------------------------------------------------

    temperature = (
        read_grib_variable(
            filename,
            VARIABLES[
                "temperature"
            ]
        )
    )


    pressure = (
        read_grib_variable(
            filename,
            VARIABLES[
                "pressure"
            ]
        )
    )


    wind_u = (
        read_grib_variable(
            filename,
            VARIABLES[
                "wind_u"
            ]
        )
    )


    wind_v = (
        read_grib_variable(
            filename,
            VARIABLES[
                "wind_v"
            ]
        )
    )


    precipitation = (
        read_grib_variable(
            filename,
            VARIABLES[
                "precipitation"
            ]
        )
    )


    # -----------------------------------------------------
    # Validate variables
    # -----------------------------------------------------

    if temperature is None:

        print(
            "Temperature not found."
        )

        return None


    if pressure is None:

        print(
            "Pressure not found."
        )

        return None


    if wind_u is None:

        print(
            "U-wind not found."
        )

        return None


    if wind_v is None:

        print(
            "V-wind not found."
        )

        return None


    if precipitation is None:

        print(
            "Precipitation not found."
        )

        return None


    # -----------------------------------------------------
    # Coordinates
    # -----------------------------------------------------

    latitudes = (
        temperature[
            "latitude"
        ]
    )

    longitudes = (
        temperature[
            "longitude"
        ]
    )


    # -----------------------------------------------------
    # Wind speed
    # -----------------------------------------------------

    wind_speed = np.sqrt(

        wind_u["values"] ** 2

        +

        wind_v["values"] ** 2

    )


    # -----------------------------------------------------
    # Create dataframe
    # -----------------------------------------------------

    data = pd.DataFrame({

        "forecast_initialization":
            metadata[
                "forecast_initialization"
            ],

        "forecast_hour":
            metadata[
                "forecast_hour"
            ],

        "forecast_day":
            metadata[
                "forecast_day"
            ],

        "forecast_day_label":
            metadata[
                "forecast_day_label"
            ],

        "forecast_time":
            metadata[
                "forecast_time"
            ],

        "latitude":
            latitudes,

        "longitude":
            longitudes,

        "forecast_temperature":
            temperature[
                "values"
            ],

        "forecast_pressure":
            pressure[
                "values"
            ],

        "forecast_wind_speed":
            wind_speed,

        "forecast_precip_rate":
            precipitation[
                "values"
            ]

    })


    return data


# =========================================================
# PROCESS ALL FILES
# =========================================================

all_data = []


print("\n")
print("=" * 70)
print("PROCESSING GFS FILES")
print("=" * 70)


for index, filename in enumerate(
    files,
    start=1
):

    print(
        f"\n[{index}/{len(files)}]"
    )


    try:

        result = process_gfs_file(
            filename
        )


        if result is not None:

            all_data.append(
                result
            )


            print(
                "Records:",
                len(result)
            )


    except Exception as e:

        print(
            "FAILED:",
            os.path.basename(
                filename
            )
        )

        print(
            "Error:",
            e
        )


# =========================================================
# CHECK DATA
# =========================================================

if len(all_data) == 0:

    raise RuntimeError(
        "\nNo valid GFS forecast data could be processed."
    )


# =========================================================
# COMBINE
# =========================================================

print("\n")
print("=" * 70)
print("COMBINING GFS DATA")
print("=" * 70)


df = pd.concat(
    all_data,
    ignore_index=True
)


# =========================================================
# SORT
# =========================================================

df = df.sort_values(

    [
        "forecast_initialization",
        "forecast_hour",
        "latitude",
        "longitude"
    ]

).reset_index(
    drop=True
)


# =========================================================
# REMOVE DUPLICATES
# =========================================================

df = df.drop_duplicates(

    subset=[

        "forecast_initialization",

        "forecast_hour",

        "latitude",

        "longitude"

    ]

).reset_index(
    drop=True
)


# =========================================================
# SAVE
# =========================================================

os.makedirs(

    os.path.dirname(
        OUTPUT_FILE
    ),

    exist_ok=True
)


df.to_csv(

    OUTPUT_FILE,

    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("GFS LOADING COMPLETED")
print("=" * 70)


print(
    "\nTotal records:",
    len(df)
)


print(
    "\nColumns:"
)

print(
    list(
        df.columns
    )
)


# =========================================================
# AVAILABLE FORECAST HOURS
# =========================================================

available_hours = sorted(

    df[
        "forecast_hour"
    ]
    .drop_duplicates()
    .tolist()

)


print(
    "\nAvailable forecast hours:"
)

print(
    available_hours
)


# =========================================================
# AVAILABLE FORECAST DAYS
# =========================================================

available_days = sorted(

    df[
        "forecast_day"
    ]
    .drop_duplicates()
    .tolist()

)


print(
    "\nAvailable forecast days:"
)

print(
    available_days
)


# =========================================================
# DAY 1 TO DAY 10 STATUS
# =========================================================

print("\n")
print("=" * 70)
print("DAY 1 - DAY 10 AVAILABILITY")
print("=" * 70)


for day in range(
    1,
    11
):

    count = len(
        df[
            df[
                "forecast_day"
            ]
            == day
        ]
    )


    if count > 0:

        hours = sorted(

            df[
                df[
                    "forecast_day"
                ]
                == day
            ][
                "forecast_hour"
            ]
            .drop_duplicates()
            .tolist()

        )

        print(
            f"Day {day:2d}: AVAILABLE | "
            f"Hours: {hours} | "
            f"Records: {count}"
        )

    else:

        print(
            f"Day {day:2d}: NOT AVAILABLE"
        )


# =========================================================
# FORECAST TIME RANGE
# =========================================================

print(
    "\nForecast time range:"
)

print(

    df[
        "forecast_time"
    ].min(),

    "to",

    df[
        "forecast_time"
    ].max()

)


# =========================================================
# INITIALIZATION RANGE
# =========================================================

print(
    "\nInitialization range:"
)

print(

    df[
        "forecast_initialization"
    ].min(),

    "to",

    df[
        "forecast_initialization"
    ].max()

)


# =========================================================
# MISSING VALUES
# =========================================================

print(
    "\nMissing values:"
)

print(
    df.isna().sum()
)


# =========================================================
# DAY-WISE RECORD COUNTS
# =========================================================

print("\n")
print("=" * 70)
print("DAY-WISE RECORD COUNTS")
print("=" * 70)


day_counts = (

    df.groupby(
        "forecast_day"
    )
    .size()
    .sort_index()

)


print(
    day_counts
)


# =========================================================
# SAVE CONFIRMATION
# =========================================================

print("\n")
print("=" * 70)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)


# =========================================================
# IMPORTANT VALIDATION MESSAGE
# =========================================================

missing_days = [

    day

    for day in range(
        1,
        11
    )

    if day not in available_days

]


if len(missing_days) > 0:

    print("\n")
    print("=" * 70)
    print("WARNING")
    print("=" * 70)

    print(
        "The loader supports Day 1-Day 10,"
        " but the downloaded GRIB files do not"
        " contain all ten forecast days."
    )

    print(
        "\nMissing days:",
        missing_days
    )

    print(
        "\nDO NOT claim these days are validated"
        " until corresponding forecast data is available."
    )

else:

    print("\n")
    print("=" * 70)
    print("DAY 1-DAY 10 DATA AVAILABLE")
    print("=" * 70)

    print(
        "All ten forecast days are present."
    )


print("\nGFS processing completed successfully!")