import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =====================
# SETTINGS
# =====================

INPUT_FILE = r"C:\Users\Tommy\CobberNeural\CobberNeural\Data Sets\Daily\Minnesota_data.csv"
OUTPUT_FILE = "weather_cleaned.csv"

# =====================
# LOAD DATA
# =====================

df = pd.read_csv(INPUT_FILE)

# =====================
# TIME FEATURE
# =====================

df["Local Time"] = pd.to_datetime(
    df["Local Time"]
)

time_feature = (
    df["Local Time"].dt.hour +
    df["Local Time"].dt.minute / 60
)

# =====================
# BUILD FEATURES
# =====================

features = pd.DataFrame()

features["Temperature"] = pd.to_numeric(
    df["Temperature (°F)"],
    errors="coerce"
)

features["DewPoint"] = pd.to_numeric(
    df["Dew Point (°F)"],
    errors="coerce"
)

features["Pressure"] = pd.to_numeric(
    df["Pressure (in)"],
    errors="coerce"
)

features["Humidity"] = pd.to_numeric(
    df["Humidity (%)"],
    errors="coerce"
)

features["WindSpeed"] = pd.to_numeric(
    df["Wind Speed (mph)"],
    errors="coerce"
)

# =====================
# RAINFALL FIX
# =====================

features["Rainfall"] = pd.to_numeric(
    df["Precipitation (in)"],
    errors="coerce"
)

# If missing, assume no rain
features["Rainfall"] = (
    features["Rainfall"]
    .fillna(0)
)

features["Time"] = time_feature

# =====================
# FILL REMAINING NaNs
# =====================

for col in features.columns:

    if features[col].isna().all():

        features[col] = 0

    else:

        features[col] = (
            features[col]
            .fillna(
                features[col].median()
            )
        )

# =====================
# SAFETY CHECK
# =====================

print("\nNaN Counts:")
print(features.isna().sum())

# =====================
# NORMALIZE
# =====================

scaler = MinMaxScaler()

features = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns
)

# =====================
# SAVE
# =====================

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)

print("\nColumns:")
print(features.columns.tolist())

print("\nShape:")
print(features.shape)