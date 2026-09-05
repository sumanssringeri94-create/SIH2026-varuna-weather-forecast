import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# FILES
# =========================================================

COMPARISON_FILE = "data/processed/forecast_comparison.csv"
FINAL_FORECAST_FILE = "data/processed/final_forecast.csv"

OUTPUT_DIR = "data/processed"


print("=" * 65)
print("WEATHER FORECAST VISUALIZATION")
print("=" * 65)


# =========================================================
# LOAD ERA5 + GFS DATA
# =========================================================

print("\nLoading ERA5 + GFS comparison data...")

comparison = pd.read_csv(
    COMPARISON_FILE
)

comparison["forecast_time"] = pd.to_datetime(
    comparison["forecast_time"]
)

print("Comparison records:", len(comparison))


# =========================================================
# LOAD ML PREDICTIONS
# =========================================================

print("\nLoading ML predictions...")

ml = pd.read_csv(
    FINAL_FORECAST_FILE
)

ml["forecast_time"] = pd.to_datetime(
    ml["forecast_time"]
)

print("ML records:", len(ml))


# =========================================================
# MERGE DATA
# =========================================================

print("\nMerging ERA5, GFS and ML data...")

df = pd.merge(
    comparison[
        [
            "forecast_time",
            "latitude",
            "longitude",
            "temperature",
            "pressure",
            "wind_speed",
            "rainfall",
            "forecast_temperature",
            "forecast_pressure",
            "forecast_wind_speed",
            "forecast_rainfall_6h"
        ]
    ],
    ml[
        [
            "forecast_time",
            "latitude",
            "longitude",
            "ml_temperature",
            "ml_pressure",
            "ml_wind_speed",
            "ml_rainfall"
        ]
    ],
    on=[
        "forecast_time",
        "latitude",
        "longitude"
    ],
    how="inner"
)


print("Merged records:", len(df))


# =========================================================
# TIME-AVERAGED REGIONAL DATA
# =========================================================

print("\nCreating regional time averages...")

plot_df = df.groupby(
    "forecast_time"
).agg({

    "temperature": "mean",
    "forecast_temperature": "mean",
    "ml_temperature": "mean",

    "pressure": "mean",
    "forecast_pressure": "mean",
    "ml_pressure": "mean",

    "wind_speed": "mean",
    "forecast_wind_speed": "mean",
    "ml_wind_speed": "mean",

    "rainfall": "mean",
    "forecast_rainfall_6h": "mean",
    "ml_rainfall": "mean"

}).reset_index()


plot_df = plot_df.sort_values(
    "forecast_time"
)


print("Forecast timestamps:", len(plot_df))

print(
    "Period:",
    plot_df["forecast_time"].min(),
    "to",
    plot_df["forecast_time"].max()
)


# =========================================================
# FUNCTION FOR PLOTTING
# =========================================================

def create_plot(
    y_columns,
    labels,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(14, 7))

    for column, label in zip(
        y_columns,
        labels
    ):

        plt.plot(
            plot_df["forecast_time"],
            plot_df[column],
            label=label,
            linewidth=2
        )

    plt.title(
        title,
        fontsize=16
    )

    plt.xlabel(
        "Date",
        fontsize=12
    )

    plt.ylabel(
        ylabel,
        fontsize=12
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output_path = (
        f"{OUTPUT_DIR}/{filename}"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# =========================================================
# TEMPERATURE
# =========================================================

create_plot(

    [
        "temperature",
        "forecast_temperature",
        "ml_temperature"
    ],

    [
        "ERA5 Observation",
        "GFS Forecast",
        "ML Prediction"
    ],

    "Temperature: ERA5 vs GFS vs ML",

    "Temperature (°C)",

    "temperature_comparison.png"
)


# =========================================================
# PRESSURE
# =========================================================

create_plot(

    [
        "pressure",
        "forecast_pressure",
        "ml_pressure"
    ],

    [
        "ERA5 Observation",
        "GFS Forecast",
        "ML Prediction"
    ],

    "Pressure: ERA5 vs GFS vs ML",

    "Pressure (hPa)",

    "pressure_comparison.png"
)


# =========================================================
# WIND SPEED
# =========================================================

create_plot(

    [
        "wind_speed",
        "forecast_wind_speed",
        "ml_wind_speed"
    ],

    [
        "ERA5 Observation",
        "GFS Forecast",
        "ML Prediction"
    ],

    "Wind Speed: ERA5 vs GFS vs ML",

    "Wind Speed (m/s)",

    "wind_speed_comparison.png"
)


# =========================================================
# RAINFALL
# =========================================================

create_plot(

    [
        "rainfall",
        "forecast_rainfall_6h",
        "ml_rainfall"
    ],

    [
        "ERA5 Observation",
        "GFS Forecast",
        "ML Prediction"
    ],

    "Rainfall: ERA5 vs GFS vs ML",

    "Rainfall (mm)",

    "rainfall_comparison.png"
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 65)
print("ALL PLOTS CREATED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated plots:")

print("1. temperature_comparison.png")
print("2. pressure_comparison.png")
print("3. wind_speed_comparison.png")
print("4. rainfall_comparison.png")