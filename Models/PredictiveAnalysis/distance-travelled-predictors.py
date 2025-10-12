import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib





# INPUT: Lon, Lat, Course, Speed, category 
# OUTPUT: Distanc travelled 





############################################
# LOAD DATA
############################################

data_path = "MR_Dataset/cleaned/"

telemetry_data = pd.read_csv(os.path.join(data_path, "telemetry_clean.csv"))

voyages_data = pd.read_csv(os.path.join(data_path, "voyages_clean.csv"))

############################################
# Merge data
############################################

# Merge telemetry and voyage to get associated ship type for each
merged = pd.merge(telemetry_data, voyages_data, on="voyage_id")

############################################
# Calculate distance travelled in next hour
############################################

# 1. Ensure timestamps
merged['observed_at'] = pd.to_datetime(merged['observed_at'])

# 2. Sort by voyage and time
merged = merged.sort_values(by=['voyage_id', 'observed_at'])

# 3. Compute distance to next ping
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

merged['dist_next'] = merged.groupby('voyage_id').apply(
    lambda g: haversine(g['lat_deg'], g['lon_deg'], g['lat_deg'].shift(-1), g['lon_deg'].shift(-1))
).reset_index(level=0, drop=True)

# 4. Compute rolling distance over next hour
def rolling_hourly_distance(group):
    group = group.set_index('observed_at').sort_index()
    group['dist_next_hour'] = group['dist_next'].rolling('1h').sum()
    return group

merged = merged.groupby('voyage_id', group_keys=False).apply(rolling_hourly_distance).reset_index(drop=True)

############################################
# SEPERATE REQUIRED DATA
############################################

# This is the same data which will be passed to it by the LLM
X = merged[["lat_deg", "lon_deg", "course_deg", "speed_kn", "category"]]

# This is Y (the type of ship)
Y = merged["dist_next_hour"]

print("Merged: ", merged[:10])
print("X: ", X[:10])
print("Y: ", Y[:10])

############################################
# Create sliding window sequences
############################################

window_size = 5  # number of past observations to use
X_seq, Y_seq = [], []

features = ["lat_deg", "lon_deg", "course_deg", "speed_kn"]  # exclude 'category' if numeric encoding not done yet
# Optional: encode 'category' numerically
X['category'] = LabelEncoder().fit_transform(X['category'])

for voyage_id, group in merged.groupby('voyage_id'):
    group = group.sort_values('observed_at')
    for i in range(len(group) - window_size):
        seq_x = group.iloc[i:i+window_size][features + ['category']].values.flatten()  # flatten 5x5 -> 25 features
        seq_y = group.iloc[i + window_size]['dist_next_hour']
        X_seq.append(seq_x)
        Y_seq.append(seq_y)

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)


############################################
# SPLIT INTO TRAIN, TEST AND VALIDATION
############################################

X_train, X_test, Y_train, Y_test = train_test_split(
    X_seq, Y_seq, test_size=0.2, random_state=42
)

############################################
# CREATE MODELS
############################################

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

############################################
# TRAIN MODELS
############################################

random_forest_model.fit(X_train, Y_train)

############################################
# MAKE PREDICTIONS
############################################

Y_hat_random_forest = random_forest_model.predict(X_test)

############################################
# EVALUTE PERFORMANCE
############################################

def evaluate_model(name, Y_test, Y_pred):
    print(f"\n=== {name} ===")
    print("MAE:", mean_absolute_error(Y_test, Y_pred))
    print("MSE:", mean_squared_error(Y_test, Y_pred))
    print("R2:", r2_score(Y_test, Y_pred))

evaluate_model("Random Forest", Y_test, Y_hat_random_forest)


############################################
# SAVE MODEL
############################################
joblib.dump(random_forest_model, "random_forest_distance_model.pkl")


