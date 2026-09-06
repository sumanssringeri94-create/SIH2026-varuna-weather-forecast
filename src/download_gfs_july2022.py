import os
import requests
from datetime import date, timedelta


# =========================================================
# CONFIGURATION
# =========================================================

OUTPUT_DIR = r"data\raw\ncmrwf_forecast"

START_DATE = date(2022, 7, 1)
END_DATE = date(2022, 7, 31)

FORECAST_HOURS = [
    6,
    24,
    48,
    72,
    96,
    120,
    144,
    168,
    192,
    216,
    240
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# DOWNLOAD ONE GFS FILE
# =========================================================

def download_gfs(
    target_date,
    forecast_hour
):

    date_string = target_date.strftime(
        "%Y%m%d"
    )

    hour_string = f"{forecast_hour:03d}"

    filename = (
        f"gfs_3_{date_string}_0000_"
        f"{hour_string}.grb2"
    )

    url = (
        f"https://www.ncei.noaa.gov/thredds/"
        f"fileServer/model-gfs-004-files/"
        f"202207/{date_string}/{filename}"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print(
        f"\nDownloading: {filename}"
    )

    print(
        f"Forecast lead: {forecast_hour} hours"
    )

    print(
        f"URL: {url}"
    )

    if os.path.exists(output_file):

        size = os.path.getsize(
            output_file
        )

        if size > 1_000_000:

            print(
                "Already exists. Skipping."
            )

            return True

        print(
            "Existing file is too small. "
            "Removing it."
        )

        os.remove(
            output_file
        )

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=300
        )

        response.raise_for_status()

        with open(
            output_file,
            "wb"
        ) as f:

            f.write(
                response.content
            )

        size = len(
            response.content
        )

        print(
            "Downloaded successfully:"
            f" {size / 1024 / 1024:.2f} MB"
        )

        return True

    except Exception as e:

        print(
            "Download failed."
        )

        print(
            "Error:",
            e
        )

        if os.path.exists(
            output_file
        ):

            os.remove(
                output_file
            )

        return False


# =========================================================
# DOWNLOAD JULY 2022
# =========================================================

print("=" * 65)
print("GFS JULY 2022 - DAY 1 TO DAY 10 DOWNLOAD")
print("=" * 65)

current_date = START_DATE

successful = 0
failed = 0

total_files = (
    (END_DATE - START_DATE).days + 1
) * len(
    FORECAST_HOURS
)

processed = 0


while current_date <= END_DATE:

    for forecast_hour in FORECAST_HOURS:

        processed += 1

        print(
            f"\n[{processed}/{total_files}]"
        )

        if download_gfs(
            current_date,
            forecast_hour
        ):

            successful += 1

        else:

            failed += 1

    current_date += timedelta(
        days=1
    )


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 65)
print("JULY DOWNLOAD SUMMARY")
print("=" * 65)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Total attempted:",
    total_files
)

print(
    "\nRequired forecast hours:"
)

print(
    FORECAST_HOURS
)

print(
    "\nGFS files currently present:"
)

files = sorted(
    f
    for f in os.listdir(
        OUTPUT_DIR
    )
    if f.endswith(".grb2")
)

for filename in files:

    size = os.path.getsize(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

    print(
        f"{filename} - "
        f"{size / 1024 / 1024:.2f} MB"
    )

print(
    "\nJuly GFS download completed."
)