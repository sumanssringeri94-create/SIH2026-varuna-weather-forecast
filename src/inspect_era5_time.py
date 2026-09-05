import cfgrib

ERA5_FILE = r"data\raw\observations\9885cbd53ccb07271e5e0f53ca219b14.grib"

print("Reading ERA5 file...")
print("File:", ERA5_FILE)

datasets = cfgrib.open_datasets(
    ERA5_FILE,
    backend_kwargs={"indexpath": ""}
)

print("\nNumber of datasets:", len(datasets))

for i, ds in enumerate(datasets):

    print("\n" + "=" * 60)
    print("DATASET", i)
    print("=" * 60)

    print("\nVariables:")
    print(list(ds.data_vars))

    print("\nDimensions:")
    print(ds.dims)

    print("\nCoordinates:")

    if "time" in ds.coords:
        print("time:")
        print(ds["time"].values)

    if "step" in ds.coords:
        print("step:")
        print(ds["step"].values)

    if "valid_time" in ds.coords:
        print("valid_time:")
        print(ds["valid_time"].values)

    print("\nData variables:")
    for var in ds.data_vars:
        print(
            var,
            "shape:",
            ds[var].shape
        )

print("\nERA5 time inspection completed.")