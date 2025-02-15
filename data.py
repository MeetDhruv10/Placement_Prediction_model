# import pandas as pd
# import numpy as np
# # Load your CSV file
# file_path = "data\\Sorted_data.csv"  # Replace with your file path
# data = pd.read_csv(file_path)

# plt.show()
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'data\\Sorted_data.csv'
data = pd.read_csv(file_path)

# Split dataset into features (X) and target (y)
X = data.drop(columns=['PlacedOrNot'])  # Replace 'PlacedOrNot' with actual target column
y = data['PlacedOrNot']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

#  Save trained column names for future use
joblib.dump(X.columns.tolist(), 'trained_columns.pkl')

# Split into train-test sets (80% train, 20% test)
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (needed for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)
# print(X_train_scaled)
# print(X_test_scaled)
# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Initialize and train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save trained KNN model
joblib.dump(knn, 'knn_model.pkl')

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
# n_estimators will create 100 decision trees in the ensemble.
#random_state used to ensure consistent and reproducible results when training your Random Forest model.
rf.fit(X_train_full, y_train)

# Save trained Random Forest model
joblib.dump(rf, 'rf_model.pkl')

# Load models back to check
knn_loaded = joblib.load('knn_model.pkl')
rf_loaded = joblib.load('rf_model.pkl')

# Predict with loaded models
y_pred_knn = knn_loaded.predict(X_test_scaled)
y_pred_rf = rf_loaded.predict(X_test_full)

# Evaluate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100

print(f"KNN Accuracy: {accuracy_knn:.2f}%")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}%")

# 🔹 Line Graph: KNN Accuracy for Different k-values
k_values = range(1, 21)
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    knn_accuracies.append(accuracy_score(y_test, y_pred_knn) * 100)

# plt.figure(figsize=(10, 5))
# plt.plot(k_values, knn_accuracies, marker='o', linestyle='-', color='blue', label='KNN Accuracy')
# plt.axhline(y=accuracy_rf, color='green', linestyle='--', label='Random Forest Accuracy')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy (%)')
# plt.title('KNN Accuracy vs. Number of Neighbors')
# plt.legend()
# plt.grid(True)
# plt.show()

# ✅ Feature Comparison: How Each Feature Affects Accuracy
features_to_test = ["Internship","Soft Skills"	,"Programming"	,"Operating Systems",	"Databases",	"Cyber Security"	,"CGPA","HistoryofBacklog"]
knn_accuracies_feature = []
rf_accuracies_feature = []

for feature in features_to_test:
    print(f"Training models without feature: {feature}")

    # Drop one feature
    X_train = X_train_full.drop(columns=[feature])
    X_test = X_test_full.drop(columns=[feature])

    # Scale for KNN
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    knn_accuracies_feature.append(accuracy_score(y_test, y_pred_knn) * 100)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracies_feature.append(accuracy_score(y_test, y_pred_rf) * 100)

# 🔹 Plot Feature Importance Effect
plt.figure(figsize=(10, 5))
plt.plot(features_to_test, knn_accuracies_feature, marker='o', linestyle='-', color='blue', label='KNN Accuracy')
plt.plot(features_to_test, rf_accuracies_feature, marker='s', linestyle='-', color='green', label='Random Forest Accuracy')

plt.xlabel("Feature Removed")
plt.ylabel("Accuracy (%)")
plt.title("Effect of Removing Features on Accuracy")
plt.legend()
plt.grid(True)
plt.show()
