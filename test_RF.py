import pandas as pd
import joblib

# Load the saved Random Forest model
rf = joblib.load("rf_model.pkl")

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

# Make predictions
rf_prediction = rf.predict(test_data)

# Convert predictions to labels
rf_result = ["Placed" if pred == 1 else "Not Placed" for pred in rf_prediction]

# Save results
test_data["Random Forest Prediction"] = rf_result

# Output the predictions
print(test_data[["Random Forest Prediction"]])

# Save results to CSV
test_data.to_csv("rf_predicted_results.csv", index=False)
print("Random Forest Predictions saved to rf_predicted_results.csv")
