import pandas as pd
import matplotlib.pyplot as plt

# Load the prediction results from both models
knn_results_path = "C:\\Users\\dhruv\\OneDrive\\Desktop\\Placement project\\knn_predicted_results.csv"
rf_results_path = "C:\\Users\\dhruv\\OneDrive\\Desktop\\Placement project\\rf_predicted_results.csv"

knn_data = pd.read_csv(knn_results_path)
rf_data = pd.read_csv(rf_results_path)

# Count the number of "Placed" and "Not Placed" predictions
knn_counts = knn_data["KNN Prediction"].value_counts()
rf_counts = rf_data["Random Forest Prediction"].value_counts()

# Create a bar chart
labels = ["Placed", "Not Placed"]
knn_values = [knn_counts.get("Placed", 0), knn_counts.get("Not Placed", 0)]
rf_values = [rf_counts.get("Placed", 0), rf_counts.get("Not Placed", 0)]

x = range(len(labels))
plt.figure(figsize=(8, 5))
plt.bar(x, knn_values, width=0.4, label="KNN", color="blue", align="center")
plt.bar([i + 0.4 for i in x], rf_values, width=0.4, label="Random Forest", color="green", align="center")

plt.xlabel("Prediction Category")
plt.ylabel("Count")
plt.title("Comparison of Placement Predictions by KNN and Random Forest")
plt.xticks([i + 0.2 for i in x], labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the graph
plt.show()
