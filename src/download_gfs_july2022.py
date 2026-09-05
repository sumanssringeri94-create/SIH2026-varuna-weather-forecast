import requests
import os
from datetime import date, timedelta

OUTPUT_DIR = r"data\raw\ncmrwf_forecast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

start_date = date(2022, 7, 1)
end_date = date(2022, 7, 31)

current = start_date

while current <= end_date:

    date_string = current.strftime("%Y%m%d")

    filename = f"gfs_3_{date_string}_0000_006.grb2"

    url = (
        f"https://www.ncei.noaa.gov/thredds/fileServer/"
        f"model-gfs-004-files/202207/{date_string}/"
        f"{filename}"
    )

    output_path = os.path.join(OUTPUT_DIR, filename)

    print("\nDownloading:", filename)
    print("URL:", url)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=300
        )

        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        print("Downloaded:", filename)
        print("Size:", len(response.content))

    except Exception as e:
        print("FAILED:", filename)
        print("Error:", e)

    current += timedelta(days=1)

print("\nJuly 2022 download completed.")