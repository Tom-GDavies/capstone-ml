import pandas as pd
import numpy as np
import os
import joblib

# -------------------------------
# Config
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_csv_path = os.path.join(script_dir, "Data", "typical_sample.csv")  # preprocessed sample
model_pkl_path = os.path.join(script_dir, "Data", "dense_autoencoder_model.pkl")  # joblib-saved model
window_size = 10

# -------------------------------
# Load preprocessed sample
# -------------------------------
df = pd.read_csv(sample_csv_path)
numeric_cols = df.select_dtypes(include=[np.number]).columns

# -------------------------------
# Sliding windows function
# -------------------------------
def create_windows(values, window_size=10, step_size=1):
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]))
    windows = []
    for i in range(0, len(values) - window_size + 1, step_size):
        windows.append(values[i:i + window_size])
    return np.array(windows)

# Create windows
sample_windows = create_windows(df[numeric_cols].values, window_size=window_size)

# Flatten for Dense autoencoder
if sample_windows.size > 0:
    X_test = sample_windows.reshape(sample_windows.shape[0], -1)
else:
    X_test = np.empty((0, window_size * len(numeric_cols)))

# -------------------------------
# Load autoencoder from .pkl
# -------------------------------
with open(model_pkl_path, "rb") as f:
    autoencoder = joblib.load(f)

# -------------------------------
# Predict and compute reconstruction error
# -------------------------------
reconstructed = autoencoder.predict(X_test)
mse = np.mean((X_test - reconstructed) ** 2, axis=1)

# Example threshold: 85th percentile of MSE
threshold = np.percentile(mse, 85)
predictions = (mse > threshold).astype(int)  # 0 = typical, 1 = anomalous

# -------------------------------
# Print results
# -------------------------------
for i, (window_mse, pred) in enumerate(zip(mse, predictions)):
    print(f"Window {i}: MSE={window_mse:.6f}, Prediction={'Anomalous' if pred==1 else 'Typical'}")
