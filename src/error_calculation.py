import pandas as pd


def calculate_errors(actual_file, forecast_file):

    actual = pd.read_csv(actual_file)
    forecast = pd.read_csv(forecast_file)

    # Combine actual and forecast using date and location
    data = pd.merge(
        actual,
        forecast,
        on=["date", "latitude", "longitude"]
    )

    # Calculate absolute errors
    data["temperature_error"] = abs(
        data["temperature"] - data["forecast_temperature"]
    )

    data["rainfall_error"] = abs(
        data["rainfall"] - data["forecast_rainfall"]
    )

    data["wind_error"] = abs(
        data["wind_speed"] - data["forecast_wind_speed"]
    )

    data["pressure_error"] = abs(
        data["pressure"] - data["forecast_pressure"]
    )

    return data

def identify_forecast_bust(data, threshold=20):
    data["forecast_bust"] = (
        data["rainfall_error"] >= threshold
    ).astype(int)

    return data


if __name__ == "__main__":

    actual_file = "data/processed/sample_weather.csv"
    forecast_file = "data/processed/sample_forecast.csv"

    result = calculate_errors(actual_file, forecast_file)

    result = identify_forecast_bust(result)

    print("\nForecast Bust Detection:")
    print(
        result[
            [
                "date",
                "rainfall_error",
                "forecast_bust"
            ]
        ]
    )