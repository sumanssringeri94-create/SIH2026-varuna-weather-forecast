import os
import requests
from datetime import date, timedelta

OUTPUT_DIR = r"data\raw\ncmrwf_forecast"

START_DATE = date(2022, 6, 3)
END_DATE = date(2022, 6, 30)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_gfs(target_date):

    date_string = target_date.strftime("%Y%m%d")

    filename = f"gfs_3_{date_string}_0000_006.grb2"

    url = (
        f"https://www.ncei.noaa.gov/thredds/fileServer/"
        f"model-gfs-004-files/202206/{date_string}/"
        f"{filename}"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print(f"\nDownloading {filename}")

    if os.path.exists(output_file):
        size = os.path.getsize(output_file)

        if size > 1000000:
            print("Already exists. Skipping.")
            return True

        os.remove(output_file)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=300
        )

        response.raise_for_status()

        with open(output_file, "wb") as f:
            f.write(response.content)

        size = len(response.content)

        print(
            f"Downloaded successfully: "
            f"{size / 1024 / 1024:.2f} MB"
        )

        return True

    except Exception as e:

        print("Download failed.")
        print("Error:", e)

        if os.path.exists(output_file):
            os.remove(output_file)

        return False


# ---------------------------------------------------------
# Download June 3 to June 30
# ---------------------------------------------------------

current_date = START_DATE

successful = 0
failed = 0

while current_date <= END_DATE:

    if download_gfs(current_date):
        successful += 1
    else:
        failed += 1

    current_date += timedelta(days=1)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

print("Successful:", successful)
print("Failed:", failed)

print("\nGFS files:")

files = sorted(
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".grb2")
)

for f in files:
    size = os.path.getsize(
        os.path.join(OUTPUT_DIR, f)
    )

    print(
        f"{f} - "
        f"{size / 1024 / 1024:.2f} MB"
    )