from pmlb import fetch_data
import pandas as pd
from GSI import calculate_GSI_DM, create_feature_map, remove_low_M_gates, pegasusQSVCTrainNoiseSimulator, pegasusQSVCTest
import numpy as np
from sklearn.model_selection import train_test_split
import time


# Load the dataset as a pandas DataFrame
df = pd.read_csv('breast-w.tsv', sep='\t')

print(df.shape)
print(df['target'].nunique())
y = df['target']
X = df.drop(['target'], axis=1)

# Split the data into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345)

# Split the training set into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=12345)
# This results in 60% training, 20% validation, 20% testing

num_qubit = X_train.shape[1]

print(f"The number of qubits for the dataset is {num_qubit}")

qc = create_feature_map(num_qubit)

numero_de_puertas = len(qc.decompose().data)
print(f"The circuit has {numero_de_puertas} gates.")

# Save the motherboard image
qc.decompose().draw('mpl', filename='./circuitsQSVM/circuit_base_BreastW_NoiseSimulator.pdf')

# Use a sample from X_train to calculate metrics
time_init = time.time()
metrics = calculate_GSI_DM(qc.decompose(), X_train.iloc[0].values)
time_end = time.time()

print(f"The time to compute the GSI is {time_end - time_init}")

for m in metrics:
    print(f"Gate: {m['gate']} at position {m['position']}")
    print(f"  Fidelity (F): {m['F']:.4f}")
    print(f"  Entanglement (E): {m['E']:.4f}")
    print(f"  (1 - Entropy) (1 - S): {1 - m['S']:.4f}")
    print(f"  (1 - Sensitivity) (1 - P): {1 - m['P']:.4f}")
    print(f"  Metric (M): {m['M']:.4f}\n")

# Extract the values ​​of the metric M
M_values = [m['M'] for m in metrics]

# Calculate the minimum and maximum of M
M_min = min(M_values)
M_max = max(M_values)

print(f"Metric (M): minimum = {M_min:.4f}, maximum = {M_max:.4f}")

# Create a list of thresholds from M_min to M_max in increments of 0.02
thresholds = np.arange(M_min, M_max + 0.02, 0.02)
thresholds = np.round(thresholds, 4)  # Round to avoid float precision issues

print(thresholds)

# Dictionaries to store results, models and circuits
results = {}
models = {}
circuits = {}

# Iterate over the thresholds and perform the process
for idx, threshold in enumerate(thresholds):
    print(f"\nEvaluating with threshold M_threshold = {threshold}")
    try:

            # Remove doors with M < threshold
            new_feature_map = remove_low_M_gates(qc, metrics, M_threshold=threshold)

            # Check the number of doors in the new circuit
            numero_de_puertas = len(new_feature_map.decompose().data)
            print(
                f"The circuit has {numero_de_puertas} gates after removing with M_threshold = {threshold}")

            # Train the model using the modified feature map
            model, score, total_time = pegasusQSVCTrainNoiseSimulator(
                X_train, y_train, X_val, y_val, new_feature_map, "0")

            print(
                f"The score of the circuit with M_threshold = {threshold} is {score} and time is {total_time}")

            # Store the results, the model and the circuit
            results[threshold] = {'Threshold': threshold, 'Score': score,
                                'Time': total_time, 'Num_Puertas': numero_de_puertas}
            models[threshold] = model
            circuits[threshold] = new_feature_map

    except Exception as e:
        print(
            f"Error al evaluar el circuito con M_threshold = {threshold}: {e}")
        # Still storing the results
        results[threshold] = {'Threshold': threshold, 'Score': None,
                              'Time': None, 'Num_Puertas': numero_de_puertas}
        models[threshold] = None
        circuits[threshold] = None

# Convert the result dictionary to a pandas DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index.name = 'Threshold'

# Export the DataFrame to a CSV file
results_df.to_csv('resultsBreastW.csv')

# Select the base circuit
print("\nSelect the base circuit...")
base_threshold = thresholds[0]
base_result = results[base_threshold]
base_accuracy = base_result['Score']
base_time = base_result['Time']
print(f"Base circuit at threshold {base_threshold} with score {base_accuracy} and time {base_time}")

# Seleccionar el mejor circuito en términos de accuracy
print("\nSelecting the best circuit in terms of accuracy...")
# Create a DataFrame without the base model
results_df_no_base = results_df.drop(index=base_threshold)

# Select the model with the highest accuracy
best_accuracy_model_info = results_df_no_base.loc[results_df_no_base['Score'].idxmax()]
best_accuracy_threshold = best_accuracy_model_info['Threshold']
best_accuracy_model = models[best_accuracy_threshold]
best_accuracy_circuit = circuits[best_accuracy_threshold]
print(
    f"Best model in terms of accuracy is at threshold {best_accuracy_threshold} with score {best_accuracy_model_info['Score']} and time{best_accuracy_model_info['Time']}")

# Save the image of the best circuit in terms of accuracy
best_accuracy_circuit.draw('mpl', filename='./circuitsQSVM/circuit_best_accuracy_BreastW_NoiseSimulator.pdf')

# Select the best circuit in terms of time
print("\nSelecting the best circuit in terms of time...")
min_accuracy = base_accuracy * 0.85  # Optional: limit to models with accuracy no more than 15% worse than the base

# Filter candidates for time
candidates_time = results_df_no_base[results_df_no_base['Score'] >= min_accuracy]

if not candidates_time.empty:
    # Select the model with the shortest time
    best_time_model_info = candidates_time.loc[candidates_time['Time'].idxmin()]
    best_time_threshold = best_time_model_info['Threshold']
    best_time_model = models[best_time_threshold]
    best_time_circuit = circuits[best_time_threshold]
    print(
        f"Best model in terms of time is at threshold {best_time_threshold} with score {best_time_model_info['Score']} and time {best_time_model_info['Time']}")

    # Save the image of the best circuit in terms of time
    best_time_circuit.draw('mpl', filename='./circuitsQSVM/circuit_best_time_BreastW_NoiseSimulator.pdf')
else:
    print("There are no models that meet the accuracy constraint to optimize time.")
    best_time_model = None
    best_time_circuit = None

# === New: Select circuit based on ranking ===
print("\nSelecting the circuit based on the custom ranking...")

# Calculate the ranking for each circuit
ranking_scores = {}

for threshold, row in results_df_no_base.iterrows():
    Ai = row['Score']
    Ti = row['Time']
    if Ai is not None and Ti is not None:
        # Calculate the ranking score according to your formula
        ranking_score = (base_accuracy - Ai) + ((base_time - Ti) / base_time)
        ranking_scores[threshold] = ranking_score
    else:
        # If there is no accuracy or time data, assign a very low ranking
        ranking_scores[threshold] = float('inf')

# Convert the ranking to a DataFrame
ranking_df = pd.DataFrame.from_dict(ranking_scores, orient='index', columns=['Ranking_Score'])
ranking_df.index.name = 'Threshold'

# Merge with original results
results_with_ranking = results_df_no_base.join(ranking_df)

print(results_with_ranking)

# Sort the circuits according to the ranking
results_with_ranking_sorted = results_with_ranking.sort_values(by='Ranking_Score', ascending=False)

# Select the circuit with the best ranking
best_ranking_threshold = results_with_ranking_sorted.index[0]
best_ranking_model_info = results_with_ranking_sorted.iloc[0]
best_ranking_model = models[best_ranking_threshold]
best_ranking_circuit = circuits[best_ranking_threshold]

print(
    f"Best ranking-based model is at threshold {best_ranking_threshold} with ranking score {best_ranking_model_info['Ranking_Score']}, score {best_ranking_model_info['Score']} and time {best_ranking_model_info['Time']}")

# Save the image of the best circuit based on ranking
best_ranking_circuit.draw('mpl', filename='./circuitsQSVM/circuit_best_ranking_BreastW_NoiseSimulator.pdf')

# === Evaluation on the test set ===

# Evaluation on the test set
print("\n=== Evaluation on the test set ===")

# Evaluate the base model on the test set
print("\nEvaluando el modelo base en el conjunto de test...")
base_model = models[base_threshold]
base_test_score, base_test_time = pegasusQSVCTest(base_model, X_test, y_test)
print(f"Accuracy of the base model on the test set: {base_test_score}")

# Evaluate the best model in terms of accuracy on the test set
if best_accuracy_model is not None:
    print("\nEvaluating the best model in terms of accuracy on the test set...")
    best_accuracy_test_score, best_accuracy_test_time = pegasusQSVCTest(best_accuracy_model, X_test, y_test)
    print(f"Accuracy of the best model in terms of accuracy on the test set: {best_accuracy_test_score}")
else:
    print("There is no best model in terms of accuracy to evaluate on the test set.")

# Evaluate the best model in terms of time on the test set
if best_time_model is not None:
    print("\nEvaluating the best model in terms of time on the test set...")
    best_time_test_score, best_time_test_time = pegasusQSVCTest(best_time_model, X_test, y_test)
    print(f"Accuracy of the best model in terms of time on the test set: {best_time_test_score}")
else:
    print("There is no best model in terms of time to evaluate on the test set.")

# Evaluate the best ranking-based model on the test set
if best_ranking_model is not None:
    print("\nEvaluating the best ranking-based model on the test set...")
    best_ranking_test_score, best_ranking_test_time = pegasusQSVCTest(best_ranking_model, X_test, y_test)
    print(f"Accuracy of the best ranking-based model on the test set: {best_ranking_test_score}")
else:
    print("There is no best ranking-based model to evaluate on the test set.")
