# import pandas as pd
# import numpy as np
# # Load your CSV file
# file_path = "data\\Sorted_data.csv"  # Replace with your file path
# data = pd.read_csv(file_path)





import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'data\\Sorted_data.csv'  # Update with the correct path
data = pd.read_csv(file_path)

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['PlacedOrNot'])  # Replace 'PlacedOrNot' with your actual target column
y = data['PlacedOrNot']

# Encode categorical features if necessary
X = pd.get_dummies(X, drop_first=True)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Important for KNN but not for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=8)  # KNN model
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest model

# Train models
knn.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)  # Random Forest does not need scaling

# Make predictions
y_pred_knn = knn.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

# Evaluate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Print dataset sizes
train_size = len(X_train)
test_size = len(X_test)
total_size = len(X)

train_percentage = (train_size / total_size) * 100
test_percentage = (test_size / total_size) * 100

print(f"Training Data: {train_percentage:.2f}% ({train_size} samples)")
print(f"Testing Data: {test_percentage:.2f}% ({test_size} samples)")

# Print results
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# --- PLOTTING THE GRAPH ---
# Model names and accuracies
models = ["KNN", "Random Forest"]
accuracies = [accuracy_knn * 100, accuracy_rf * 100]  # Convert to percentage

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.bar(models, accuracies, color=['blue', 'green'])

# Add labels and title
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.title("KNN vs Random Forest Accuracy")
plt.ylim(0, 100)  # Set Y-axis limit from 0 to 100%
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the accuracy values on top of the bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)

# Show the plot
plt.show()
