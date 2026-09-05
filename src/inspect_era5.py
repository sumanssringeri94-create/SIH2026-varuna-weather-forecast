import eccodes


FILE_PATH = (
    "data/raw/observations/"
    "9885cbd53ccb07271e5e0f53ca219b14.grib"
)


print("Inspecting ERA5 GRIB messages...\n")


with open(FILE_PATH, "rb") as f:

    count = 0

    while True:

        gid = eccodes.codes_grib_new_from_file(f)

        if gid is None:
            break

        count += 1

        try:

            short_name = eccodes.codes_get(
                gid,
                "shortName"
            )

            name = eccodes.codes_get(
                gid,
                "name"
            )

            type_of_level = eccodes.codes_get(
                gid,
                "typeOfLevel"
            )

            try:
                level = eccodes.codes_get(
                    gid,
                    "level"
                )
            except Exception:
                level = "N/A"

            data_date = eccodes.codes_get(
                gid,
                "dataDate"
            )

            data_time = eccodes.codes_get(
                gid,
                "dataTime"
            )

            print(
                count,
                "|",
                short_name,
                "|",
                type_of_level,
                "| level:",
                level,
                "| date:",
                data_date,
                "| time:",
                data_time,
                "|",
                name
            )

        except Exception as e:

            print(
                count,
                "| ERROR:",
                e
            )

        finally:

            eccodes.codes_release(gid)


print("\nTotal messages:", count)