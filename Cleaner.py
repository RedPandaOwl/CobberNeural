import pandas as pd
import numpy as np

# 1. Load the original CSV file
df = pd.read_csv("data/minneapolis.csv")

# Record the original number of rows
original_rows = len(df)

# 2. Convert the date into numeric features
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfYear"] = df["Date"].dt.dayofyear

# Remove the original date column
df = df.drop(columns=["Date"])

# 3. Identify the weather measurement columns
weather_columns = [
    "Maximum Temperature degrees (F)",
    "Minimum Temperature degrees (F)",
    "Precipitation (inches)",
    "Snow (inches)",
    "Snow Depth (inches)"
]

# 4. Convert weather measurements to numbers
for col in weather_columns:
    # Treat trace measurements as zero for now
    df[col] = df[col].replace("T", "0")

    # Treat M and empty entries as missing
    df[col] = df[col].replace(
        {"M": np.nan, "": np.nan}
    )

    # Convert values to numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Remove every row containing missing data
df = df.dropna()

# 6. Ensure all remaining columns are numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Remove any rows that became missing during conversion
df = df.dropna()

# 7. Save the cleaned dataset
df.to_csv("weather_cleaned.csv", index=False)

# 8. Display results
print("Cleaning complete!")
print(f"Original rows: {original_rows}")
print(f"Rows remaining: {len(df)}")
print(f"Rows removed: {original_rows - len(df)}")
print(f"Columns remaining: {len(df.columns)}")
print(f"Missing values remaining: {df.isna().sum().sum()}")
print("\nFirst five rows:")
print(df.head())

print("\nSaved as weather_cleaned.csv")