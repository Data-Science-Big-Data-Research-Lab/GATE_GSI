from pmlb import fetch_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from GSI import calculate_GSI, create_feature_map, remove_low_M_gates, pegasusQSVCTrain, pegasusQSVCTest

# Load the dataset as a pandas DataFrame
df = fetch_data('breast_w')

# Display dataset information
print(f"Dataset shape: {df.shape}")
print(f"Number of unique target labels: {df['target'].nunique()}")

# Separate features (X) and target (y)
y = df['target']
X = df.drop(['target'], axis=1)

# Split the data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=12345
)
# Result: 60% training, 20% validation, 20% testing

# Determine the number of qubits
num_qubits = X_train.shape[1]
print(f"The number of qubits for the dataset is {num_qubits}")

# Create the initial feature map
qc = create_feature_map(num_qubits)

# Get the number of gates in the circuit
num_gates = len(qc.decompose().data)
print(f"The circuit has {num_gates} gates.")

qc.decompose().draw('mpl', filename='circuitsQSVM/circuit_base_breastW.png')
print("Base circuit image saved as 'circuitsQSVM/circuit_base_breastW.png'.")

# Use a sample from X_train to calculate the metrics
start_time = time.time()
metrics = calculate_GSI(qc.decompose(), X_train.iloc[0].values)
end_time = time.time()
print(f"Time to compute GSI: {end_time - start_time} seconds")

# Extract M values
M_values = [m['M'] for m in metrics]

# Calculate min and max of M
M_min = min(M_values)
M_max = max(M_values)
print(f"GSI: min = {M_min:.4f}, max = {M_max:.4f}")

# Generate thresholds from M_min to M_max in increments of 0.02
thresholds = np.arange(M_min, M_max + 0.02, 0.02)
thresholds = np.round(thresholds, 4)

print(f"Thresholds: {thresholds}")

# Initialize dictionaries to store results, models, and circuits
results = {}
models = {}
circuits = {}

# Iterate through thresholds and process each
for idx, threshold in enumerate(thresholds):
    print(f"\nEvaluating with M_threshold = {threshold}")
    try:
        # Remove gates with GSI < threshold
        new_feature_map = remove_low_M_gates(qc, metrics, M_threshold=threshold)

        # Check the number of gates in the new circuit
        num_gates = len(new_feature_map.decompose().data)
        print(f"The circuit has {num_gates} gates after removing with M_threshold = {threshold}")

        # Train the model using the modified feature map
        model, score, total_time = pegasusQSVCTrain(X_train, y_train, X_val, y_val, new_feature_map)

        print(f"The score for M_threshold = {threshold} is {score}, and the time is {total_time} seconds")

        # Store the results, model, and circuit
        results[threshold] = {'Threshold': threshold, 'Score': score, 'Time': total_time, 'Num_Gates': num_gates}
        models[threshold] = model
        circuits[threshold] = new_feature_map

    except Exception as e:
        print(f"Error encountered at M_threshold = {threshold}: {e}")
        print("Exiting loop, a qubit has not a gate.")
        break  # Exit the loop when a qubit has not a gate

# Convert results dictionary to a pandas DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index.name = 'Threshold'

# Export the DataFrame to a CSV file
results_df.to_csv('resultsBreastWQSVM.csv')
print("\nResults saved to 'resultsBreastWQSVM.csv'.")

# Select and display base circuit details
print("\nSelecting the base circuit...")
base_threshold = thresholds[0]
base_result = results[base_threshold]
base_accuracy = base_result['Score']
base_time = base_result['Time']
print(f"Base circuit at threshold {base_threshold} with score {base_accuracy} and time {base_time} seconds.")

# Select and display the best circuit by accuracy
print("\nSelecting the best circuit by accuracy...")
results_df_no_base = results_df.drop(index=base_threshold)
best_accuracy_model_info = results_df_no_base.loc[results_df_no_base['Score'].idxmax()]
best_accuracy_threshold = best_accuracy_model_info['Threshold']
best_accuracy_model = models[best_accuracy_threshold]
best_accuracy_circuit = circuits[best_accuracy_threshold]
print(f"Best model by accuracy at threshold {best_accuracy_threshold} with score {best_accuracy_model_info['Score']}.")

# Save the best accuracy circuit image
best_accuracy_circuit.draw('mpl', filename='circuitsQSVM/circuit_best_accuracy_breastW.png')
print("Best accuracy circuit image saved as 'circuitsQSVM/circuit_best_accuracy_breastW.png'.")

# Select and display the best circuit by time
print("\nSelecting the best circuit by time...")
min_accuracy = base_accuracy * 0.85  # Allow a margin of 15% worse than base accuracy
candidates_time = results_df_no_base[results_df_no_base['Score'] >= min_accuracy]

if not candidates_time.empty:
    best_time_model_info = candidates_time.loc[candidates_time['Time'].idxmin()]
    best_time_threshold = best_time_model_info['Threshold']
    best_time_model = models[best_time_threshold]
    best_time_circuit = circuits[best_time_threshold]
    print(f"Best model by time at threshold {best_time_threshold} with score {best_time_model_info['Score']}.")

    # Save the best time circuit image
    best_time_circuit.draw('mpl', filename='circuitsQSVM/circuit_best_time_breastW.png')
    print("Best time circuit image saved as 'circuitsQSVM/circuit_best_time_breastW.png'.")
else:
    print("No models meet the accuracy constraint for time optimization.")
    best_time_model = None
    best_time_circuit = None

# === Select the circuit based on ranking ===
print("\nSelecting the circuit based on custom ranking...")

# Calculate the ranking score for each circuit
ranking_scores = {}

for threshold, row in results_df_no_base.iterrows():
    Ai = row['Score']  # Accuracy of the model
    Ti = row['Time']   # Training time of the model
    if Ai is not None and Ti is not None:
        # Calculate the ranking score based on the custom formula
        ranking_score = (Ai - base_accuracy) + ((base_time - Ti) / base_time)
        ranking_scores[threshold] = ranking_score
    else:
        # Assign a very low ranking score if accuracy or time is missing
        ranking_scores[threshold] = float('-inf')  # Worst possible ranking

# Convert the ranking scores to a DataFrame
ranking_df = pd.DataFrame.from_dict(ranking_scores, orient='index', columns=['Ranking_Score'])
ranking_df.index.name = 'Threshold'

# Combine the ranking scores with the original results
results_with_ranking = results_df_no_base.join(ranking_df)

print("\nResults with ranking:")
print(results_with_ranking)

# Sort the circuits by their ranking score
results_with_ranking_sorted = results_with_ranking.sort_values(by='Ranking_Score', ascending=False)

# Select the circuit with the best ranking score
best_balanced_threshold = results_with_ranking_sorted.index[0]
best_balanced_model_info = results_with_ranking_sorted.iloc[0]
best_balanced_model = models[best_balanced_threshold]
best_balanced_circuit = circuits[best_balanced_threshold]

print(
    f"Best model based on ranking is at threshold {best_balanced_threshold} "
    f"with ranking score {best_balanced_model_info['Ranking_Score']}, "
    f"accuracy {best_balanced_model_info['Score']}, "
    f"and time {best_balanced_model_info['Time']} seconds."
)

# Save the best ranking circuit image
best_balanced_circuit.draw('mpl', filename='circuitsQSVM/circuit_best_ranking_breastW.png')
print("Best ranking circuit image saved as 'circuitsQSVM/circuit_best_ranking_breastW.png'.")

# Evaluate the selected models on the test set
print("\n=== Test Set Evaluation ===")

# Evaluate the base model
print("\nEvaluating the base model on the test set...")
base_test_score, base_test_time = pegasusQSVCTest(models[base_threshold], X_test, y_test)
print(f"Base model test accuracy: {base_test_score}")

# Evaluate the best accuracy model
if best_accuracy_model is not None:
    print("\nEvaluating the best accuracy model on the test set...")
    best_accuracy_test_score, best_accuracy_test_time = pegasusQSVCTest(best_accuracy_model, X_test, y_test)
    print(f"Best accuracy model test accuracy: {best_accuracy_test_score}")

# Evaluate the best time model
if best_time_model is not None:
    print("\nEvaluating the best time model on the test set...")
    best_time_test_score, best_time_test_time = pegasusQSVCTest(best_time_model, X_test, y_test)
    print(f"Best time model test accuracy: {best_time_test_score}")

# Evaluate the best model balanced on the test set
if best_balanced_model is not None:
    print("\nEvaluating the best model based on ranking on the test set...")
    best_ranking_test_score, best_ranking_test_time = pegasusQSVCTest(best_balanced_model, X_test, y_test)
    print(f"Test accuracy of the best model based on ranking: {best_ranking_test_score}")
else:
    print("No best model based on ranking available for evaluation on the test set.")