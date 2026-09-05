import eccodes

FILE_PATH = r"data\raw\ncmrwf_forecast\gfs_3_20220601_0000_006.grb2"

print("Inspecting GFS forecast times...")
print()

f = open(FILE_PATH, "rb")

times = {}
total_messages = 0

while True:

    gid = eccodes.codes_grib_new_from_file(f)

    if gid is None:
        break

    total_messages += 1

    data_date = eccodes.codes_get(gid, "dataDate")
    data_time = eccodes.codes_get(gid, "dataTime")
    forecast_time = eccodes.codes_get(gid, "forecastTime")

    key = (
        data_date,
        data_time,
        forecast_time
    )

    if key not in times:
        times[key] = 0

    times[key] += 1

    eccodes.codes_release(gid)


f.close()


print("Total GRIB messages:")
print(total_messages)

print()
print("Forecast time groups:")
print("=" * 50)

for key, count in sorted(times.items()):

    data_date, data_time, forecast_time = key

    print(
        f"Date: {data_date} | "
        f"Time: {data_time:04d} | "
        f"Forecast hour: {forecast_time} | "
        f"Messages: {count}"
    )

print()
print("Number of unique forecast groups:")
print(len(times))