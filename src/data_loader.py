import pandas as pd


def load_weather_data(file_path):
    data = pd.read_csv(file_path)

    print("Weather data loaded successfully!")
    print("Number of rows:", len(data))
    print("Columns:", list(data.columns))

    return data


if __name__ == "__main__":
    file_path = "data/processed/sample_weather.csv"

    weather_data = load_weather_data(file_path)

    print("\nFirst 5 records:")
    print(weather_data.head())