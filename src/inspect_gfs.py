import eccodes

FILE = r"data\raw\ncmrwf_forecast\gfs_3_20220601_0000_006.grb2"

f = open(FILE, "rb")

variables = set()
count = 0

while True:
    handle = eccodes.codes_grib_new_from_file(f)

    if handle is None:
        break

    short_name = eccodes.codes_get(handle, "shortName")
    level_type = eccodes.codes_get(handle, "typeOfLevel")

    variables.add((short_name, level_type))
    count += 1

    eccodes.codes_release(handle)

f.close()

print("Total GRIB messages:", count)
print("\nAvailable variables:")

for variable in sorted(variables):
    print(variable)