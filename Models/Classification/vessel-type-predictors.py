import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder





# INPUT: Lon, Lat, Course, Speed
# OUTPUT: Typical or normal





############################################
# LOAD DATA
############################################

data_path = "MR_Dataset/cleaned/"

telemetry_data = pd.read_csv(os.path.join(data_path, "telemetry_clean.csv"))

voyages_data = pd.read_csv(os.path.join(data_path, "voyages_clean.csv"))

############################################
# SEPERATE REQUIRED DATA
############################################

# Merge telemetry and voyage to get associated ship type for each
merged = pd.merge(telemetry_data, voyages_data, on="voyage_id")

# This is the same data which will be passed to it by the LLM
X = merged[["lat_deg", "lon_deg", "course_deg", "speed_kn"]]

# This is Y (the type of ship)
Y = merged["category"]


print("X: ", X[:10])
print("Y: ", Y[:10])

############################################
# SPLIT INTO TRAIN, TEST AND VALIDATION
############################################

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)


############################################
# APPLY SMOTE TO BALANCE TRAINING DATA
############################################

smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

print("Before SMOTE class distribution:\n", Y_train.value_counts())
print("After SMOTE class distribution:\n", pd.Series(Y_train_res).value_counts())

############################################
# ENCODE FOR XGBOOST
############################################

le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train_res)  # e.g., 'cargo' -> 0
Y_test_encoded = le.transform(Y_test)

############################################
# CREATE MODELS
############################################

logistic_regression_model = LogisticRegression(
    max_iter=2000, 
    solver="lbfgs",
    class_weight="balanced"
)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)

decision_tree_model = DecisionTreeClassifier(
    max_depth=None,
    class_weight="balanced",
    random_state=42
)


xgb_model = xgb.XGBClassifier( 
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.1, 
    objective="multi:softmax", 
    num_class=len(le.classes_), 
    random_state=42)


############################################
# TRAIN MODELS
############################################

logistic_regression_model.fit(X_train_res, Y_train_res)
knn_model.fit(X_train_res, Y_train_res)
decision_tree_model.fit(X_train_res, Y_train_res)
xgb_model.fit(X_train_res, Y_train_encoded)

############################################
# MAKE PREDICTIONS
############################################

Y_hat_logistic_regression = logistic_regression_model.predict(X_test)
Y_hat_knn = knn_model.predict(X_test)
Y_hat_decision_tree = decision_tree_model.predict(X_test)
Y_hat_xgboost_encoded = xgb_model.predict(X_test)
Y_hat_xgboost = le.inverse_transform(Y_hat_xgboost_encoded)


############################################
# EVALUTE PERFORMANCE
############################################

def evaluate_model(name, Y_test, Y_pred, classes):
    print(f"\n=== {name} ===")
    acc = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print("Confusion Matrix:\n", 
          pd.DataFrame(confusion_matrix(Y_test, Y_pred),
                       index=classes,
                       columns=classes))

evaluate_model("Logistic Regression", Y_test, Y_hat_logistic_regression, logistic_regression_model.classes_)
evaluate_model("KNN", Y_test, Y_hat_knn, knn_model.classes_)
evaluate_model("Decision Tree", Y_test, Y_hat_decision_tree, decision_tree_model.classes_)
evaluate_model("XGBoost", Y_test, Y_hat_xgboost, xgb_model.classes_)

############################################
# SAVE MODEL
############################################

joblib.dump(logistic_regression_model, "logistic_regression_ship_type.pkl")
joblib.dump(knn_model, "knn_ship_type.pkl")
joblib.dump(decision_tree_model, "decision_tree_ship_type.pkl")
joblib.dump(xgb_model, "xgboost_ship_type.pkl")
