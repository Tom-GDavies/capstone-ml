import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ---------------------------
# Config
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "Data")
typical_path = os.path.join(DATA_DIR, "typical_windows.npy")
anomalous_path = os.path.join(DATA_DIR, "anomalous_windows.npy")

MODEL_PATH = os.path.join(DATA_DIR, "dense_autoencoder_model.keras")
H5_PATH = os.path.join(DATA_DIR, "dense_autoencoder_model.h5")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

train = True

# ---------------------------
# Load data
# ---------------------------
typical = np.load(typical_path)
anomalous = np.load(anomalous_path)

# If data has 3 dims, take only the latest timestep (make it 2D)
if typical.ndim == 3:
    typical = typical[:, -1, :]
    anomalous = anomalous[:, -1, :]

print("Typical shape:", typical.shape)
print("Anomalous shape:", anomalous.shape)

# ---------------------------
# Scale data
# ---------------------------
scaler = StandardScaler()
X_typical = scaler.fit_transform(typical)
X_anomalous = scaler.transform(anomalous)
joblib.dump(scaler, SCALER_PATH)

# ---------------------------
# Train/Val/Test split (on typical)
# ---------------------------
X_train, X_temp = train_test_split(X_typical, test_size=0.3, random_state=42)
X_val, X_typical_test = train_test_split(X_temp, test_size=0.5, random_state=42)
X_anomalous_test = X_anomalous

# ---------------------------
# Dense Autoencoder
# ---------------------------
def create_dense_autoencoder(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(input_dim, activation='linear') 
    ])
    model.compile(optimizer=Adam(1e-4), loss=Huber())
    return model


input_dim = X_train.shape[1]

if train:
    autoencoder = create_dense_autoencoder(input_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(
        X_train, X_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=1
    )
    autoencoder.save(MODEL_PATH)
    autoencoder.save(H5_PATH)
else:
    autoencoder = load_model(MODEL_PATH)

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_model(model, X_test):
    reconstructed = model.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructed), axis=1)
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
