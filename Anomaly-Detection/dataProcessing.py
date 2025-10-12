import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load data
# -------------------------------
typical_df = pd.read_csv("Data/typical.csv")
anomalous_df = pd.read_csv("Data/anomalous.csv")

# Combine datasets if desired
df = pd.concat([typical_df, anomalous_df], ignore_index=True)

# -------------------------------
# 2. Convert TIMESTAMP to time difference
# -------------------------------
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")

# Sort by Track_ID and TIMESTAMP
df = df.sort_values(by=["Track_ID", "TIMESTAMP"])

# Compute time difference in seconds within each Track_ID
df['time_diff'] = df.groupby("Track_ID")['TIMESTAMP'].diff().dt.total_seconds().fillna(0)

# -------------------------------
# 3. Drop Track_ID and original TIMESTAMP
# -------------------------------
df = df.drop(columns=["Track_ID", "TIMESTAMP"])

# -------------------------------
# 4. Normalise all numeric columns
# -------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------
# 5. Sort by index again if needed
# -------------------------------
df = df.reset_index(drop=True)

# -------------------------------
# 6. Create overlapping windows
# -------------------------------
def create_windows(values, window_size=10, step_size=1):
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]))  # empty 3D array
    windows = []
    for i in range(0, len(values) - window_size + 1, step_size):
        windows.append(values[i:i + window_size])
    return np.array(windows)

all_windows = []

for track_id, group in pd.concat([typical_df[['Track_ID']], df], axis=1).groupby("Track_ID"):
    values = group[numeric_cols].values
    windows = create_windows(values, window_size=10, step_size=1)
    if windows.size > 0:  # skip empty windows
        all_windows.append(windows)

if all_windows:
    X = np.vstack(all_windows)
else:
    X = np.empty((0, 10, len(numeric_cols)))  # fallback

# -------------------------------
# 7. Combine into one array
# -------------------------------
X = np.vstack(all_windows)  # shape = (num_windows, 10, num_features)

print("Final data shape:", X.shape)

np.save("Data/windowed.npy", X)
