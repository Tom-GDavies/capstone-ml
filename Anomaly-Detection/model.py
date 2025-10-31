import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ---------------------------
# Config
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "Data")
typical_path = os.path.join(DATA_DIR, "typical_windows.npy")
anomalous_path = os.path.join(DATA_DIR, "anomalous_windows.npy")

MODEL_PATH = os.path.join(DATA_DIR, "autoencoder_model.keras")   # Keras format
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")              # sklearn scaler

train = True

# ---------------------------
# Load windows
# ---------------------------
typical_windows = np.load(typical_path)
anomalous_windows = np.load(anomalous_path)

print("Typical:", typical_windows.shape, "Anomalous:", anomalous_windows.shape)

# ---------------------------
# Ensure temporal length divisible by 8 (3 pooling layers -> 2**3)
# ---------------------------
def pad_to_multiple(arr, multiple=8):
    # arr shape: (n_windows, window_size, n_features)
    w = arr.shape[1]
    rem = w % multiple
    if rem == 0:
        return arr
    pad_len = multiple - rem
    pad_shape = ((0, 0), (0, pad_len), (0, 0))
    return np.pad(arr, pad_shape, mode='edge')  # pad by repeating last row

typical_windows = pad_to_multiple(typical_windows, multiple=8)
anomalous_windows = pad_to_multiple(anomalous_windows, multiple=8)

# ---------------------------
# Train/Val/Test split (on typical)
# ---------------------------
X_train, X_temp = train_test_split(typical_windows, test_size=0.3, random_state=42)
X_val, X_typical_test = train_test_split(X_temp, test_size=0.5, random_state=42)
X_anomalous_test = anomalous_windows

# ---------------------------
# Optionally: fit scaler on flattened features and apply (if you didn't already)
# If windows are already scaled, skip this. Example shown for completeness.
# ---------------------------
# Flatten windows to 2D for scaler: (num_rows, features)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# flat_train = X_train.reshape(-1, X_train.shape[2])
# scaler.fit(flat_train)
# # apply scaler per-window
# def apply_scaler_to_windows(windows, scaler):
#     flat = windows.reshape(-1, windows.shape[2])
#     flat = scaler.transform(flat)
#     return flat.reshape(windows.shape)
# X_train = apply_scaler_to_windows(X_train, scaler)
# X_val = apply_scaler_to_windows(X_val, scaler)
# X_typical_test = apply_scaler_to_windows(X_typical_test, scaler)
# X_anomalous_test = apply_scaler_to_windows(X_anomalous_test, scaler)
# joblib.dump(scaler, SCALER_PATH)

# ---------------------------
# Model creation (final layer activation = linear)
# ---------------------------
def create_autoencoder(input_shape):
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, padding='same'))
    # decoder
    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(UpSampling1D(2))
    # reconstruction -- linear for scaled continuous data
    model.add(Conv1D(input_shape[1], 3, activation='linear', padding='same'))
    model.compile(optimizer=Adam(1e-4), loss=Huber())
    return model

input_shape = (typical_windows.shape[1], typical_windows.shape[2])

if train:
    autoencoder = create_autoencoder(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, X_val),
                    callbacks=[early_stopping])
    # save Keras model (recommended, NOT pickling)
    autoencoder.save(MODEL_PATH)
else:
    autoencoder = load_model(MODEL_PATH)

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_model(model, X_test):
    reconstructed = model.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
    return mse

typical_mse = evaluate_model(autoencoder, X_typical_test)
anomalous_mse = evaluate_model(autoencoder, X_anomalous_test)

y_true = np.concatenate([np.zeros(len(typical_mse)), np.ones(len(anomalous_mse))])
all_mse = np.concatenate([typical_mse, anomalous_mse])
threshold = np.percentile(typical_mse, 85)
y_pred = (all_mse > threshold).astype(int)

print(f"Chosen threshold: {threshold:.6f}")
print("Overall Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["Typical", "Anomalous"]))
