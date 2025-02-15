import pandas as pd
import joblib

# Load the saved KNN model and scaler
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")  # Load the scaler

# Load new test data
test_file = "data\\test_data.csv"  # Replace with your actual test file
test_data = pd.read_csv(test_file)

# Process test data (Encode categorical variables)
test_data = pd.get_dummies(test_data)

# Ensure test data has the same columns as training data
trained_features = joblib.load("trained_columns.pkl")  # Load feature columns from training
for col in trained_features:
    if col not in test_data.columns:
        test_data[col] = 0  # Add missing columns

# Scale test data (KNN requires scaling)
test_data_scaled = scaler.transform(test_data)

# Make predictions
knn_prediction = knn.predict(test_data_scaled)

# Convert predictions to labels
knn_result = ["Placed" if pred == 1 else "Not Placed" for pred in knn_prediction]

# Save results
test_data["KNN Prediction"] = knn_result

# Output the predictions
print(test_data[["KNN Prediction"]])

# Save results to CSV
test_data.to_csv("knn_predicted_results.csv", index=False)
print("KNN Predictions saved to knn_predicted_results.csv")
