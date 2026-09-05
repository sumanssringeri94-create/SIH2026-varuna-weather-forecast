import eccodes

FILE_PATH = r"data\raw\ncmrwf_forecast\gfs_3_20220601_0000_006.grb2"

f = open(FILE_PATH, "rb")

count = 0
found = []

while True:
    gid = eccodes.codes_grib_new_from_file(f)

    if gid is None:
        break

    count += 1

    short_name = eccodes.codes_get(gid, "shortName")
    level_type = eccodes.codes_get(gid, "typeOfLevel")

    if short_name in ["10u", "10v", "2t", "sp", "prate"]:
        level = eccodes.codes_get(gid, "level")

        found.append(
            (short_name, level_type, level)
        )

    eccodes.codes_release(gid)

f.close()

print("Total GRIB messages:", count)

print("\nRequired variables found:")

for item in sorted(set(found)):
    print(item)