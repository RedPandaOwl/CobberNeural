import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Make results more reproducible
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------
# 1. Load the cleaned weather data
# -----------------------------------------

df = pd.read_csv("data/minneapolis_weather_cleaned.csv")

# Reconstruct the date from Year and DayOfYear
df["Date"] = pd.to_datetime(
    df["Year"].astype(str)
    + df["DayOfYear"].astype(str).str.zfill(3),
    format="%Y%j"
)

# Sort observations chronologically
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------------------
# 2. Create tomorrow's temperature target
# -----------------------------------------

target_column = "Maximum Temperature degrees (F)"

# Make a table of tomorrow's temperatures,
# indexed by the date they should be predicted from.
targets = df[["Date", target_column]].copy()

targets["Date"] = targets["Date"] - pd.Timedelta(days=1)

targets = targets.rename(
    columns={target_column: "TomorrowMaxTemp"}
)

# Match each day's weather with the next calendar day's temperature.
# This avoids accidentally treating two nonconsecutive rows as
# consecutive days when missing observations were removed.
df = df.merge(targets, on="Date", how="inner")

# -----------------------------------------
# 3. Choose the input features
# -----------------------------------------

feature_columns = [
    "Maximum Temperature degrees (F)",
    "Minimum Temperature degrees (F)",
    "Precipitation (inches)",
    "Snow (inches)",
    "Snow Depth (inches)",
    "Month",
    "DayOfYear"
]

X = df[feature_columns].to_numpy(dtype=np.float32)
y = df["TomorrowMaxTemp"].to_numpy(dtype=np.float32)

print("Number of usable examples:", len(df))
print("Number of input features:", len(feature_columns))

# -----------------------------------------
# 4. Split chronologically
# -----------------------------------------

# Older data for training, newer data for evaluation.
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

if train_end == 0 or val_end <= train_end or val_end >= n:
    raise ValueError(
        "Not enough usable observations for train/validation/test splits."
    )

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# -----------------------------------------
# 5. Scale the input features
# -----------------------------------------

# Learn scaling parameters from training data only.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# -----------------------------------------
# 6. Build the neural network
# -----------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(feature_columns),)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

model.summary()

# -----------------------------------------
# 7. Train the network
# -----------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------------------
# 8. Evaluate on unseen test data
# -----------------------------------------

predictions = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, predictions)

print("\nTest results")
print("------------")
print(f"Mean absolute error: {mae:.2f} degrees F")

# Compare predictions with actual temperatures.
results = pd.DataFrame({
    "Date": df["Date"].iloc[val_end:].to_numpy(),
    "Actual_Temperature": y_test,
    "Predicted_Temperature": predictions
})

print("\nSample predictions:")
print(results.head(10).to_string(index=False))

# Save predictions and the trained model.
results.to_csv("weather_predictions.csv", index=False)
model.save("weather_model.keras")

print("\nSaved model to weather_model.keras")
print("Saved predictions to weather_predictions.csv")