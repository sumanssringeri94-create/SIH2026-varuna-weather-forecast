import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# FILE PATHS
# =========================================================

ML_FILE = "data/processed/ml_predictions_improved.csv"
BUST_FILE = "data/processed/forecast_bust_results.csv"

OUTPUT_DIR = "data/processed/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("FINAL WEATHER FORECAST EVALUATION")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading ML predictions...")

ml = pd.read_csv(ML_FILE)

ml["forecast_time"] = pd.to_datetime(
    ml["forecast_time"]
)

print("ML records:", len(ml))


print("\nLoading forecast bust results...")

bust = pd.read_csv(BUST_FILE)

bust["forecast_time"] = pd.to_datetime(
    bust["forecast_time"]
)

print("Bust records:", len(bust))


# =========================================================
# MERGE DATA
# =========================================================

print("\nMerging datasets...")

merge_columns = [
    "forecast_time",
    "latitude",
    "longitude"
]

bust_columns = merge_columns + [
    "bust_score",
    "forecast_bust",
    "dominant_factor"
]

bust_small = bust[bust_columns].copy()

df = pd.merge(
    ml,
    bust_small,
    on=merge_columns,
    how="inner"
)

print("Merged records:", len(df))


# =========================================================
# MODEL PERFORMANCE
# =========================================================

targets = {
    "Temperature": {
        "actual": "temperature",
        "gfs": "forecast_temperature",
        "ml": "ml_temperature"
    },

    "Pressure": {
        "actual": "pressure",
        "gfs": "forecast_pressure",
        "ml": "ml_pressure"
    },

    "Wind Speed": {
        "actual": "wind_speed",
        "gfs": "forecast_wind_speed",
        "ml": "ml_wind_speed"
    },

    "Rainfall": {
        "actual": "rainfall",
        "gfs": "forecast_rainfall_6h",
        "ml": "ml_rainfall"
    }
}


results = []


print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


for name, columns in targets.items():

    actual = df[columns["actual"]]

    gfs = df[columns["gfs"]]

    ml_prediction = df[columns["ml"]]


    # -----------------------------------------------------
    # GFS metrics
    # -----------------------------------------------------

    gfs_mae = mean_absolute_error(
        actual,
        gfs
    )

    gfs_rmse = np.sqrt(
        mean_squared_error(
            actual,
            gfs
        )
    )

    gfs_r2 = r2_score(
        actual,
        gfs
    )


    # -----------------------------------------------------
    # ML metrics
    # -----------------------------------------------------

    ml_mae = mean_absolute_error(
        actual,
        ml_prediction
    )

    ml_rmse = np.sqrt(
        mean_squared_error(
            actual,
            ml_prediction
        )
    )

    ml_r2 = r2_score(
        actual,
        ml_prediction
    )


    # -----------------------------------------------------
    # Improvement
    # -----------------------------------------------------

    improvement = (
        (gfs_mae - ml_mae)
        / gfs_mae
        * 100
        if gfs_mae != 0
        else 0
    )


    results.append({

        "Variable": name,

        "GFS_MAE": gfs_mae,
        "ML_MAE": ml_mae,

        "GFS_RMSE": gfs_rmse,
        "ML_RMSE": ml_rmse,

        "GFS_R2": gfs_r2,
        "ML_R2": ml_r2,

        "Improvement_Percent": improvement

    })


    print(f"\n{name}")

    print(
        f"GFS MAE : {gfs_mae:.4f}"
    )

    print(
        f"ML MAE  : {ml_mae:.4f}"
    )

    print(
        f"GFS RMSE: {gfs_rmse:.4f}"
    )

    print(
        f"ML RMSE : {ml_rmse:.4f}"
    )

    print(
        f"GFS R²  : {gfs_r2:.4f}"
    )

    print(
        f"ML R²   : {ml_r2:.4f}"
    )

    print(
        f"Improvement: {improvement:.2f}%"
    )


performance_df = pd.DataFrame(results)


# =========================================================
# SAVE PERFORMANCE TABLE
# =========================================================

performance_file = (
    f"{OUTPUT_DIR}/model_performance.csv"
)

performance_df.to_csv(
    performance_file,
    index=False
)

print("\nSaved:")
print(performance_file)


# =========================================================
# BUST RISK SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("FORECAST BUST RISK SUMMARY")
print("=" * 70)


risk_counts = (
    df["forecast_bust"]
    .value_counts()
    .reindex(
        [
            "NORMAL",
            "WATCH",
            "WARNING",
            "CRITICAL"
        ],
        fill_value=0
    )
)


print("\nRisk levels:")

print(risk_counts)


risk_percent = (
    risk_counts
    / len(df)
    * 100
)


print("\nRisk percentages:")

for level in risk_counts.index:

    print(
        f"{level:10s}: "
        f"{risk_counts[level]:4d} "
        f"({risk_percent[level]:.2f}%)"
    )


# =========================================================
# DOMINANT FACTORS
# =========================================================

print("\n")
print("=" * 70)
print("DOMINANT WEATHER FACTORS")
print("=" * 70)


factor_counts = (
    df["dominant_factor"]
    .value_counts()
)


print(
    factor_counts
)


# =========================================================
# BUST SCORE STATISTICS
# =========================================================

print("\n")
print("=" * 70)
print("BUST SCORE STATISTICS")
print("=" * 70)


print(
    df["bust_score"].describe()
)


# =========================================================
# CRITICAL FORECASTS
# =========================================================

critical = (
    df[
        df["forecast_bust"] == "CRITICAL"
    ]
    .sort_values(
        "bust_score",
        ascending=False
    )
)


print("\n")
print("=" * 70)
print("TOP CRITICAL FORECASTS")
print("=" * 70)


critical_columns = [
    "forecast_time",
    "latitude",
    "longitude",
    "bust_score",
    "forecast_bust",
    "dominant_factor"
]


print(
    critical[
        critical_columns
    ]
    .head(20)
    .to_string(index=False)
)


critical_file = (
    f"{OUTPUT_DIR}/critical_forecasts.csv"
)

critical[
    critical_columns
].to_csv(
    critical_file,
    index=False
)

print("\nSaved:")
print(critical_file)


# =========================================================
# PLOT 1 — MODEL MAE
# =========================================================

plt.figure(
    figsize=(12, 6)
)

x = np.arange(
    len(performance_df)
)

width = 0.35

plt.bar(
    x - width / 2,
    performance_df["GFS_MAE"],
    width,
    label="GFS"
)

plt.bar(
    x + width / 2,
    performance_df["ML_MAE"],
    width,
    label="ML"
)

plt.xticks(
    x,
    performance_df["Variable"]
)

plt.ylabel("Mean Absolute Error")

plt.title(
    "GFS vs ML Forecast Accuracy"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

mae_plot = (
    f"{OUTPUT_DIR}/gfs_vs_ml_mae.png"
)

plt.savefig(
    mae_plot,
    dpi=300
)

plt.close()

print("\nSaved:")
print(mae_plot)


# =========================================================
# PLOT 2 — IMPROVEMENT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    performance_df["Variable"],
    performance_df["Improvement_Percent"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel(
    "MAE Improvement (%)"
)

plt.title(
    "ML Improvement over GFS"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

improvement_plot = (
    f"{OUTPUT_DIR}/ml_improvement.png"
)

plt.savefig(
    improvement_plot,
    dpi=300
)

plt.close()

print("Saved:")
print(improvement_plot)


# =========================================================
# PLOT 3 — RISK DISTRIBUTION
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    risk_counts.index,
    risk_counts.values
)

plt.xlabel(
    "Forecast Risk Level"
)

plt.ylabel(
    "Number of Forecasts"
)

plt.title(
    "Forecast Bust Risk Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

risk_plot = (
    f"{OUTPUT_DIR}/risk_distribution.png"
)

plt.savefig(
    risk_plot,
    dpi=300
)

plt.close()

print("Saved:")
print(risk_plot)


# =========================================================
# PLOT 4 — DOMINANT FACTORS
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    factor_counts.index,
    factor_counts.values
)

plt.xlabel(
    "Dominant Weather Factor"
)

plt.ylabel(
    "Number of Forecasts"
)

plt.title(
    "Dominant Factors Causing Forecast Deviations"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

factor_plot = (
    f"{OUTPUT_DIR}/dominant_factors.png"
)

plt.savefig(
    factor_plot,
    dpi=300
)

plt.close()

print("Saved:")
print(factor_plot)


# =========================================================
# REGIONAL RISK
# =========================================================

print("\n")
print("=" * 70)
print("HIGHEST-RISK LOCATIONS")
print("=" * 70)


location_risk = (
    df.groupby(
        [
            "latitude",
            "longitude"
        ]
    )
    .agg(
        average_bust_score=(
            "bust_score",
            "mean"
        ),

        maximum_bust_score=(
            "bust_score",
            "max"
        ),

        critical_count=(
            "forecast_bust",
            lambda x:
            (x == "CRITICAL").sum()
        ),

        warning_count=(
            "forecast_bust",
            lambda x:
            (x == "WARNING").sum()
        )
    )
    .reset_index()
)


location_risk = (
    location_risk
    .sort_values(
        "average_bust_score",
        ascending=False
    )
)


print(
    location_risk
    .head(20)
    .to_string(index=False)
)


location_file = (
    f"{OUTPUT_DIR}/location_risk.csv"
)

location_risk.to_csv(
    location_file,
    index=False
)

print("\nSaved:")
print(location_file)


# =========================================================
# DAILY RISK TREND
# =========================================================

daily_risk = (
    df.assign(
        date=df["forecast_time"].dt.date
    )
    .groupby("date")
    .agg(
        average_bust_score=(
            "bust_score",
            "mean"
        ),

        maximum_bust_score=(
            "bust_score",
            "max"
        ),

        critical_count=(
            "forecast_bust",
            lambda x:
            (x == "CRITICAL").sum()
        )
    )
    .reset_index()
)


daily_file = (
    f"{OUTPUT_DIR}/daily_risk.csv"
)

daily_risk.to_csv(
    daily_file,
    index=False
)


# =========================================================
# PLOT 5 — DAILY BUST SCORE
# =========================================================

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    daily_risk["date"],
    daily_risk["average_bust_score"],
    marker="o",
    label="Average Bust Score"
)

plt.plot(
    daily_risk["date"],
    daily_risk["maximum_bust_score"],
    marker="o",
    label="Maximum Bust Score"
)

plt.xlabel("Date")

plt.ylabel("Bust Score")

plt.title(
    "Daily Forecast Bust Risk"
)

plt.legend()

plt.grid(True)

plt.xticks(
    rotation=45
)

plt.tight_layout()

daily_plot = (
    f"{OUTPUT_DIR}/daily_bust_risk.png"
)

plt.savefig(
    daily_plot,
    dpi=300
)

plt.close()

print("Saved:")
print(daily_plot)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION COMPLETED")
print("=" * 70)

print("\nDataset records:", len(df))

print(
    "Forecast period:",
    df["forecast_time"].min(),
    "to",
    df["forecast_time"].max()
)

print(
    "\nCritical forecasts:",
    risk_counts["CRITICAL"]
)

print(
    "Warning forecasts:",
    risk_counts["WARNING"]
)

print(
    "Watch forecasts:",
    risk_counts["WATCH"]
)

print(
    "Normal forecasts:",
    risk_counts["NORMAL"]
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. model_performance.csv")
print("2. critical_forecasts.csv")
print("3. location_risk.csv")
print("4. daily_risk.csv")
print("5. gfs_vs_ml_mae.png")
print("6. ml_improvement.png")
print("7. risk_distribution.png")
print("8. dominant_factors.png")
print("9. daily_bust_risk.png")

print("\nAll evaluation analysis completed successfully!")