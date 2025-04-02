import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = 'data\\Sorted_data.csv'
data = pd.read_csv(file_path)

# Split dataset into features (X) and target (y)
X = data.drop(columns=['PlacedOrNot'])  
y = data['PlacedOrNot']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Save trained column names
joblib.dump(X.columns.tolist(), 'trained_columns.pkl')

# Split into train-test sets
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)
joblib.dump(scaler, 'scaler.pkl')

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_accuracy = accuracy_score(y_test, knn.predict(X_test_scaled)) * 100
joblib.dump(knn, 'knn_model.pkl')

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_full, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test_full)) * 100
joblib.dump(rf, 'rf_model.pkl')

# 📌 Graph 1: KNN Accuracy vs. Number of Neighbors
# k_values = range(1, 21)
# knn_accuracies = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train).predict(X_test_scaled)) * 100 for k in k_values]

# plt.figure(figsize=(10, 5))
# plt.plot(k_values, knn_accuracies, marker='o', linestyle='-', color='blue', label='KNN Accuracy')
# plt.axhline(y=rf_accuracy, color='green', linestyle='--', label='Random Forest Accuracy')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy (%)')
# plt.title('KNN Accuracy vs. Number of Neighbors')
# plt.legend()
# plt.grid(True)
# plt.show()

# 📌 Graph 2: Bar Chart - Accuracy Comparison
plt.figure(figsize=(6, 5))
plt.bar(["KNN", "Random Forest"], [knn_accuracy, rf_accuracy], color=['blue', 'green'])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.show()

# 📌 Graph 3: Feature Importance (Random Forest)
importances = rf.feature_importances_
feature_names = X_train_full.columns

plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color='green')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (Random Forest)")
plt.show()

# 📌 Graph 4: Heatmap - Feature Correlations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 📌 Graph 5: Box Plot - CGPA Distribution for Placement
plt.figure(figsize=(8, 5))
sns.boxplot(x=y, y=data["CGPA"])
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("CGPA")
plt.title("CGPA Distribution for Placed vs. Not Placed Students")
plt.show()
