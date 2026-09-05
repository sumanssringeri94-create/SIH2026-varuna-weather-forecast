import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VARUNA | Weather Forecast Bust Detection",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1350px;
        }

        .brand {
            font-size: 3.2rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            margin-bottom: 0.1rem;
        }

        .tagline {
            color: #aeb4c0;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }

        .pipeline {
            background: #151922;
            border: 1px solid #2a303b;
            border-radius: 10px;
            padding: 14px 18px;
            color: #d9dde5;
            font-size: 0.98rem;
            margin: 1rem 0 1.5rem 0;
        }

        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #173d2b;
            color: #67d39b;
            border: 1px solid #285d43;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .section-divider {
            margin: 28px 0;
            border-top: 1px solid #2b3039;
        }

        [data-testid="stMetric"] {
            background: #151922;
            border: 1px solid #292f39;
            border-radius: 10px;
            padding: 12px 14px;
        }

        [data-testid="stMetricLabel"] {
            color: #aeb4c0;
        }

        [data-testid="stMetricValue"] {
            font-weight: 650;
        }

        .sidebar-brand {
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sidebar-note {
            color: #9da4b0;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .alert-box {
            padding: 14px 16px;
            border-radius: 9px;
            margin: 8px 0 18px 0;
        }

        .alert-critical {
            background: #3b1c22;
            border: 1px solid #74323f;
            color: #ffb3be;
        }

        .alert-normal {
            background: #153325;
            border: 1px solid #285a41;
            color: #72d7a0;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "forecast_bust_results.csv"
PERFORMANCE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "evaluation"
    / "model_performance.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

if not DATA_PATH.exists():
    st.error(
        f"Forecast dataset not found:\n\n`{DATA_PATH}`\n\n"
        "Run the forecast bust detection pipeline first."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "forecast_time",
    "latitude",
    "longitude",
    "forecast_temperature",
    "forecast_pressure",
    "forecast_wind_speed",
    "forecast_rainfall_6h",
    "ml_temperature",
    "ml_pressure",
    "ml_wind_speed",
    "ml_rainfall",
    "temperature_correction",
    "pressure_correction",
    "wind_correction",
    "rainfall_correction",
    "abs_temperature_correction",
    "abs_pressure_correction",
    "abs_wind_correction",
    "abs_rainfall_correction",
    "bust_score",
    "forecast_bust",
    "dominant_factor",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The forecast dataset is missing required columns:\n\n"
        + "\n".join(f"- {col}" for col in missing_columns)
    )
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["forecast_time"] = pd.to_datetime(
    df["forecast_time"],
    errors="coerce",
)

df = df.dropna(subset=["forecast_time"]).copy()

df["forecast_date"] = df["forecast_time"].dt.date

df["forecast_bust"] = (
    df["forecast_bust"]
    .astype(str)
    .str.upper()
)

df["dominant_factor"] = (
    df["dominant_factor"]
    .astype(str)
    .str.upper()
)

df["bust_score"] = pd.to_numeric(
    df["bust_score"],
    errors="coerce",
)

risk_order = [
    "NORMAL",
    "WATCH",
    "WARNING",
    "CRITICAL",
]

risk_colors = {
    "NORMAL": "#66c2a5",
    "WATCH": "#e6b84d",
    "WARNING": "#ff8c42",
    "CRITICAL": "#e63946",
}

factor_order = [
    "TEMPERATURE",
    "PRESSURE",
    "WIND",
    "RAINFALL",
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">VARUNA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-note">'
        "Weather forecast correction and bust-risk analysis"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Forecast Controls")

    available_dates = sorted(
        df["forecast_date"].dropna().unique()
    )

    if not available_dates:
        st.error("No valid forecast dates are available.")
        st.stop()

    selected_date = st.selectbox(
        "Forecast Date",
        available_dates,
        index=0,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
    )

    st.markdown("---")

    st.markdown("### Risk Levels")
    st.markdown(
        """
        **NORMAL** — low deviation

        **WATCH** — moderate deviation

        **WARNING** — significant deviation

        **CRITICAL** — high deviation
        """
    )

    st.markdown("---")

    st.markdown(
        f"**Forecast range**  \n"
        f"{df['forecast_date'].min()} to {df['forecast_date'].max()}"
    )

    st.caption(f"{len(df):,} forecast records")


# ============================================================
# FILTER SELECTED DATE
# ============================================================

daily_df = df[
    df["forecast_date"] == selected_date
].copy()

if daily_df.empty:
    st.warning(
        f"No forecast data is available for {selected_date}."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        '<div class="brand">VARUNA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="tagline">'
        "AI-Based Weather Forecast Bust Detection"
        "</div>",
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;">'
        '<span class="status">System Ready</span>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="pipeline">'
    "<strong>GFS Forecast</strong>"
    " &nbsp;→&nbsp; "
    "<strong>ML Correction</strong>"
    " &nbsp;→&nbsp; "
    "<strong>Bust Risk Detection</strong>"
    "</div>",
    unsafe_allow_html=True,
)

st.write(
    "VARUNA compares numerical weather forecasts with "
    "machine-learning corrected values and identifies locations "
    "where the forecast may significantly deviate."
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# FORECAST OVERVIEW
# ============================================================

st.header(f"Forecast Overview — {selected_date}")

total_points = len(daily_df)

risk_counts = (
    daily_df["forecast_bust"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
)

critical_count = int(risk_counts["CRITICAL"])
warning_count = int(risk_counts["WARNING"])
watch_count = int(risk_counts["WATCH"])
normal_count = int(risk_counts["NORMAL"])

average_score = daily_df["bust_score"].mean()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Forecast Points", total_points)

with c2:
    st.metric("Critical", critical_count)

with c3:
    st.metric("Warning", warning_count)

with c4:
    st.metric("Watch", watch_count)

with c5:
    st.metric("Average Bust Score", f"{average_score:.3f}")


# ============================================================
# RISK MAP
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("Forecast Bust Risk Map")

st.caption(
    "Each point represents a forecast location. "
    "Point size indicates bust-score magnitude."
)

map_df = daily_df.copy()

map_df["forecast_bust"] = pd.Categorical(
    map_df["forecast_bust"],
    categories=risk_order,
    ordered=True,
)

fig_map = px.scatter_map(
    map_df,
    lat="latitude",
    lon="longitude",
    color="forecast_bust",
    size="bust_score",
    size_max=30,
    zoom=4.5,
    center={
        "lat": float(map_df["latitude"].mean()),
        "lon": float(map_df["longitude"].mean()),
    },
    map_style="open-street-map",
    color_discrete_map=risk_colors,
    category_orders={
        "forecast_bust": risk_order
    },
    hover_name="forecast_bust",
    hover_data={
        "latitude": ":.2f",
        "longitude": ":.2f",
        "bust_score": ":.3f",
        "dominant_factor": True,
        "ml_temperature": ":.2f",
        "ml_pressure": ":.2f",
        "ml_wind_speed": ":.2f",
        "ml_rainfall": ":.2f",
    },
)

fig_map.update_layout(
    height=560,
    margin=dict(l=0, r=0, t=0, b=0),
    legend_title_text="Risk Level",
)

st.plotly_chart(
    fig_map,
    use_container_width=True,
)


# ============================================================
# RISK DISTRIBUTION + DOMINANT FACTOR
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.header("Risk Distribution")

    risk_plot_df = risk_counts.reset_index()
    risk_plot_df.columns = [
        "Risk Level",
        "Locations",
    ]

    fig_risk = px.bar(
        risk_plot_df,
        x="Risk Level",
        y="Locations",
        text="Locations",
        category_orders={
            "Risk Level": risk_order
        },
        color="Risk Level",
        color_discrete_map=risk_colors,
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        height=420,
        showlegend=False,
        margin=dict(t=25, b=20),
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )


with right:
    st.header("Dominant Bust Factor")

    factor_counts = (
        daily_df["dominant_factor"]
        .value_counts()
        .reindex(factor_order, fill_value=0)
        .reset_index()
    )

    factor_counts.columns = [
        "Weather Variable",
        "Locations",
    ]

    fig_factor = px.pie(
        factor_counts,
        names="Weather Variable",
        values="Locations",
        hole=0.48,
    )

    fig_factor.update_traces(
        textinfo="percent+label"
    )

    fig_factor.update_layout(
        height=420,
        margin=dict(t=25, b=20),
    )

    st.plotly_chart(
        fig_factor,
        use_container_width=True,
    )


# ============================================================
# DAILY RISK TREND
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("Daily Forecast Bust Trend")

trend_df = (
    df.groupby(
        ["forecast_date", "forecast_bust"],
        observed=False,
    )
    .size()
    .reset_index(name="Forecast Points")
)

trend_df["forecast_date"] = pd.to_datetime(
    trend_df["forecast_date"]
)

fig_trend = px.bar(
    trend_df,
    x="forecast_date",
    y="Forecast Points",
    color="forecast_bust",
    category_orders={
        "forecast_bust": risk_order
    },
    color_discrete_map=risk_colors,
)

fig_trend.update_layout(
    barmode="stack",
    height=480,
    xaxis_title="Date",
    yaxis_title="Forecast Points",
    legend_title="Risk Level",
)

st.plotly_chart(
    fig_trend,
    use_container_width=True,
)


# ============================================================
# GFS VS ML COMPARISON
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("GFS Forecast vs ML-Corrected Forecast")

variable = st.selectbox(
    "Weather Variable",
    [
        "Temperature",
        "Pressure",
        "Wind Speed",
        "Rainfall",
    ],
)

variable_config = {
    "Temperature": {
        "forecast": "forecast_temperature",
        "ml": "ml_temperature",
        "unit": "°C",
    },
    "Pressure": {
        "forecast": "forecast_pressure",
        "ml": "ml_pressure",
        "unit": "hPa",
    },
    "Wind Speed": {
        "forecast": "forecast_wind_speed",
        "ml": "ml_wind_speed",
        "unit": "m/s",
    },
    "Rainfall": {
        "forecast": "forecast_rainfall_6h",
        "ml": "ml_rainfall",
        "unit": "mm",
    },
}

config = variable_config[variable]

comparison = daily_df[
    [
        "latitude",
        "longitude",
        config["forecast"],
        config["ml"],
    ]
].copy()

comparison["Location"] = (
    comparison["latitude"].map(lambda x: f"{x:.0f}°N")
    + " / "
    + comparison["longitude"].map(lambda x: f"{x:.0f}°E")
)

comparison = comparison.sort_values(
    ["latitude", "longitude"]
).reset_index(drop=True)

comparison["GFS Forecast"] = comparison[
    config["forecast"]
]

comparison["ML Corrected"] = comparison[
    config["ml"]
]

comparison_long = comparison[
    [
        "Location",
        "GFS Forecast",
        "ML Corrected",
    ]
].melt(
    id_vars="Location",
    var_name="Forecast Type",
    value_name="Value",
)

fig_comparison = px.bar(
    comparison_long,
    x="Location",
    y="Value",
    color="Forecast Type",
    barmode="group",
)

fig_comparison.update_layout(
    height=500,
    xaxis_title="Forecast Location",
    yaxis_title=f"{variable} ({config['unit']})",
    legend_title="Forecast Type",
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True,
)


# ============================================================
# CRITICAL ALERTS
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("Critical Forecast Alerts")

critical_df = daily_df[
    daily_df["forecast_bust"] == "CRITICAL"
].copy()

if critical_df.empty:
    st.markdown(
        '<div class="alert-box alert-normal">'
        "No critical forecast locations detected for this date."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="alert-box alert-critical">'
        f"<strong>{len(critical_df)} critical location(s)</strong> "
        "require attention."
        "</div>",
        unsafe_allow_html=True,
    )

    critical_display = critical_df[
        [
            "forecast_time",
            "latitude",
            "longitude",
            "bust_score",
            "dominant_factor",
            "temperature_correction",
            "pressure_correction",
            "wind_correction",
            "rainfall_correction",
        ]
    ].sort_values(
        "bust_score",
        ascending=False,
    )

    st.dataframe(
        critical_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# HIGHEST-RISK LOCATIONS
# ============================================================

st.header("Highest-Risk Locations")

highest_risk = daily_df[
    [
        "latitude",
        "longitude",
        "bust_score",
        "forecast_bust",
        "dominant_factor",
    ]
].sort_values(
    "bust_score",
    ascending=False,
).head(10).copy()

highest_risk.columns = [
    "Latitude",
    "Longitude",
    "Bust Score",
    "Risk Level",
    "Dominant Factor",
]

highest_risk["Bust Score"] = highest_risk[
    "Bust Score"
].round(4)

st.dataframe(
    highest_risk,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ML-CORRECTED WEATHER SUMMARY
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("ML-Corrected Weather Summary")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Temperature",
        f"{daily_df['ml_temperature'].mean():.2f} °C",
    )

with s2:
    st.metric(
        "Pressure",
        f"{daily_df['ml_pressure'].mean():.2f} hPa",
    )

with s3:
    st.metric(
        "Wind Speed",
        f"{daily_df['ml_wind_speed'].mean():.2f} m/s",
    )

with s4:
    st.metric(
        "Rainfall",
        f"{daily_df['ml_rainfall'].mean():.2f} mm",
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if PERFORMANCE_PATH.exists():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    with st.expander("Model Performance"):
        performance_df = pd.read_csv(PERFORMANCE_PATH)

        st.caption(
            "Performance values generated by the final evaluation pipeline."
        )

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FORECAST DATASET
# ============================================================

with st.expander("View Forecast Dataset"):
    display_columns = [
        "forecast_time",
        "latitude",
        "longitude",
        "forecast_temperature",
        "forecast_pressure",
        "forecast_wind_speed",
        "forecast_rainfall_6h",
        "ml_temperature",
        "ml_pressure",
        "ml_wind_speed",
        "ml_rainfall",
        "bust_score",
        "forecast_bust",
        "dominant_factor",
    ]

    st.dataframe(
        daily_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ABOUT VARUNA
# ============================================================

with st.expander("About VARUNA"):
    st.markdown(
        """
        **VARUNA** is a weather forecast bust-detection system.

        **Pipeline**

        GFS Forecast → Machine Learning Correction → Bust Risk Detection

        **Data**
        - GFS numerical weather forecasts
        - ERA5 observations used during model development and evaluation

        **Machine Learning**
        - Random Forest based correction models

        **Variables**
        - Temperature
        - Atmospheric pressure
        - Wind speed
        - Rainfall

        **Risk Levels**
        - NORMAL
        - WATCH
        - WARNING
        - CRITICAL

        The bust score measures the magnitude of deviation between
        the original forecast and the ML-corrected forecast.
        The dominant factor identifies the weather variable contributing
        most strongly to that deviation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.caption(
    "VARUNA | GFS + ERA5 + Random Forest Machine Learning"
)