import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_weather_data(input_file, output_file):
    """
    Cleans weather data for neural network training.
    Works with either weather dataset format.
    """

    # ==========================
    # Load Data
    # ==========================

    df = pd.read_csv(r"C:\Users\Tommy\CobberNeural\CobberNeural\Data Sets\Daily\Minnesota_data.csv")

    # ==========================
    # Parse Timestamp
    # ==========================

    df["Local Time"] = pd.to_datetime(df["Local Time"])

    # Time Features
    hour = (
        df["Local Time"].dt.hour +
        df["Local Time"].dt.minute / 60
    )

    day_of_year = df["Local Time"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    # ==========================
    # Wind Direction
    # ==========================

    direction_map = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5,
        "CALM": 0
    }

    df["Wind Angle"] = (
        df["Wind Direction"]
        .map(direction_map)
        .fillna(0)
    )

    angle_rad = np.deg2rad(df["Wind Angle"])

    df["wind_dir_sin"] = np.sin(angle_rad)
    df["wind_dir_cos"] = np.cos(angle_rad)

    # ==========================
    # Numeric Features
    # ==========================

    numeric_cols = [
        "Temperature (°F)",
        "Dew Point (°F)",
        "Humidity (%)",
        "Wind Speed (mph)",
        "Wind Gust (mph)",
        "Pressure (in)",
        "Precipitation (in)"
    ]

    for col in numeric_cols:

        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        median = df[col].median()

        if pd.isna(median):
            median = 0

        df[col] = df[col].fillna(median)

    # ==========================
    # Weather Conditions
    # ==========================

    if "Condition" in df.columns:

        condition_dummies = pd.get_dummies(
            df["Condition"],
            prefix="Condition"
        )

        df = pd.concat(
            [df, condition_dummies],
            axis=1
        )

    # ==========================
    # Build Feature Matrix
    # ==========================

    columns_to_keep = []

    for col in numeric_cols:
        if col in df.columns:
            columns_to_keep.append(col)

    columns_to_keep.extend([
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "wind_dir_sin",
        "wind_dir_cos"
    ])

    condition_cols = [
        c for c in df.columns
        if c.startswith("Condition_")
    ]

    columns_to_keep.extend(condition_cols)

    features = df[columns_to_keep].copy()

    # ==========================
    # Normalize
    # ==========================

    scaler = MinMaxScaler()

    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )

    # ==========================
    # Save
    # ==========================

    features_scaled.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved cleaned data to {output_file}")
    print("\nShape:", features_scaled.shape)
    print("\nColumns:")
    print(features_scaled.columns.tolist())


# ======================================
# Example Usage
# ======================================

clean_weather_data(
    "weather.csv",
    "weather_cleaned.csv"
)