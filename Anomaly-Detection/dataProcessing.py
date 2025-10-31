import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# -------------------------------
# 1. Load data
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
typical_df = pd.read_csv(os.path.join(script_dir, "Data", "typical.csv"))
anomalous_df = pd.read_csv(os.path.join(script_dir, "Data", "anomalous.csv"))

# -------------------------------
# 2. Preprocessing function (shared)
# -------------------------------
def preprocess(df):
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by=["Track_ID", "TIMESTAMP"])
    df['time_diff'] = df.groupby("Track_ID")['TIMESTAMP'].diff().dt.total_seconds().fillna(0)
    df = df.drop(columns=["Track_ID", "TIMESTAMP"])
    df = df.reset_index(drop=True)
    return df

# -------------------------------
# 3. Process both datasets
# -------------------------------
typical_df_proc = preprocess(typical_df)
anomalous_df_proc = preprocess(anomalous_df)

# -------------------------------
# 4. Normalise (fit on typical, transform both)
# -------------------------------
numeric_cols = typical_df_proc.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
typical_df_proc[numeric_cols] = scaler.fit_transform(typical_df_proc[numeric_cols])
anomalous_df_proc[numeric_cols] = scaler.transform(anomalous_df_proc[numeric_cols])

# -------------------------------
# 5. Windowing function
# -------------------------------
def create_windows(values, window_size=10, step_size=1):
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]))
    windows = []
    for i in range(0, len(values) - window_size + 1, step_size):
        windows.append(values[i:i + window_size])
    return np.array(windows)

# -------------------------------
# 6. Create windows per Track_ID
# -------------------------------
def make_all_windows(original_df, processed_df):
    all_windows = []
    for track_id, group in pd.concat([original_df[['Track_ID']], processed_df], axis=1).groupby("Track_ID"):
        values = group[numeric_cols].values
        windows = create_windows(values, window_size=10, step_size=1)
        if windows.size > 0:
            all_windows.append(windows)
    return np.vstack(all_windows) if all_windows else np.empty((0, 10, len(numeric_cols)))

typical_windows = make_all_windows(typical_df, typical_df_proc)
anomalous_windows = make_all_windows(anomalous_df, anomalous_df_proc)

# -------------------------------
# 7. Output results
# -------------------------------
print("Typical windows shape:", typical_windows.shape)
print("Anomalous windows shape:", anomalous_windows.shape)

np.save(os.path.join(script_dir, "Data", "typical_windows.npy"), typical_windows)
np.save(os.path.join(script_dir, "Data", "anomalous_windows.npy"), anomalous_windows)
