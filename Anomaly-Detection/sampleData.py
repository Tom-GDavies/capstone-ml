import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Paths
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "Data")

TYPICAL_PATH = os.path.join(DATA_DIR, "typical_windows.npy")
ANOMALOUS_PATH = os.path.join(DATA_DIR, "anomalous_windows.npy")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
OUTPUT_SAMPLE = os.path.join(DATA_DIR, "mixed_sample.csv")

# -------------------------------
# Config
# -------------------------------
n_total = 100           # total rows in sample
n_typical = n_total // 2
n_anomalous = n_total - n_typical

# -------------------------------
# Load windows
# -------------------------------
typical_windows = np.load(TYPICAL_PATH)
anomalous_windows = np.load(ANOMALOUS_PATH)

# Use last timestep to match model input
if typical_windows.ndim == 3:
    typical_windows = typical_windows[:, -1, :]
    anomalous_windows = anomalous_windows[:, -1, :]

# -------------------------------
# Sample rows
# -------------------------------
typical_sample = typical_windows[np.random.choice(len(typical_windows), n_typical, replace=True)]
anomalous_sample = anomalous_windows[np.random.choice(len(anomalous_windows), n_anomalous, replace=True)]

# -------------------------------
# Load scaler and scale samples
# -------------------------------
scaler = joblib.load(SCALER_PATH)
typical_sample_scaled = scaler.transform(typical_sample)
anomalous_sample_scaled = scaler.transform(anomalous_sample)

# -------------------------------
# Create DataFrames
# -------------------------------
columns = ['CRAFT_ID','LON','LAT','COURSE','SPEED','time_diff']
typical_df = pd.DataFrame(typical_sample_scaled, columns=columns)
anomalous_df = pd.DataFrame(anomalous_sample_scaled, columns=columns)

# -------------------------------
# Combine and shuffle
# -------------------------------
mixed_df = pd.concat([typical_df, anomalous_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------
# Save CSV
# -------------------------------
mixed_df.to_csv(OUTPUT_SAMPLE, index=False)
print(f"Mixed sample CSV saved to: {OUTPUT_SAMPLE}")
print(mixed_df.head(10))  # show first 10 rows for verification
