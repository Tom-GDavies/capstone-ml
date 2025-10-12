import os
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import Huber
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train or load the model
train = True

# Load in data
typical_windows = np.load('typical_windows.npy')
anomalous_windows = np.load('anomalous_windows.npy')

print(f"Typical windows shape: {typical_windows.shape}")
print(f"Anomalous windows shape: {anomalous_windows.shape}")

# Split typical data into training, validation, and typical test sets
X_train, X_temp = train_test_split(typical_windows, test_size=0.3, random_state=42)
X_val, X_typical_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# Anomalous data used only for testing
X_anomalous_test = anomalous_windows

# Define the autoencoder
def create_autoencoder(input_shape):
    model = Sequential()

    # Encoder
    model.add(Conv1D(128, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, padding='same'))

    # Decoder
    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(UpSampling1D(2))

    # Reconstruction layer
    model.add(Conv1D(input_shape[1], 3, activation='sigmoid', padding='same'))

    model.compile(optimizer=Adam(1e-4), loss=Huber())
    return model

input_shape = (typical_windows.shape[1], typical_windows.shape[2])

if train:
    autoencoder = create_autoencoder(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping]
    )
    autoencoder.save('autoencoder_model.keras')
else:
    autoencoder = load_model('autoencoder_model.keras')

# Evaluate the model
def evaluate_model(model, X_test):
    reconstructed = model.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
    return mse

typical_mse = evaluate_model(autoencoder, X_typical_test)
anomalous_mse = evaluate_model(autoencoder, X_anomalous_test)

# Combine true labels and predictions
y_true = np.concatenate([np.zeros(len(typical_mse)), np.ones(len(anomalous_mse))])
all_mse = np.concatenate([typical_mse, anomalous_mse])

# Threshold: 85th percentile of typical errors
threshold = np.percentile(typical_mse, 85)
y_pred = (all_mse > threshold).astype(int)

# Overall accuracy
overall_accuracy = accuracy_score(y_true, y_pred)
print(f"Chosen threshold: {threshold:.4f}")
print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Confusion matrix and per-class metrics
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Typical", "Anomalous"]))
