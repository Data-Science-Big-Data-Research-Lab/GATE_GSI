from pmlb import fetch_data
import pandas as pd
from tool import calculate_gate_metrics, create_feature_map, remove_low_M_gates, pegasusQSVCTrain, pegasusQSVCTest
import numpy as np
from sklearn.model_selection import train_test_split
import time

# Cargar el dataset como un DataFrame de pandas
df = fetch_data('corral')

print(df.shape)
print(df['target'].nunique())
y = df['target']
X = df.drop(['target'], axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345)

# Dividir el conjunto de entrenamiento en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=12345)
# Esto resulta en 60% entrenamiento, 20% validación, 20% prueba

num_qubit = X_train.shape[1]

print(f"The number of qubits for the dataset is {num_qubit}")

qc = create_feature_map(num_qubit)

numero_de_puertas = len(qc.decompose().data)
print(f"The circuit has {numero_de_puertas} gates.")

# Guardar la imagen del circuito base
qc.decompose().draw('mpl', filename='circuitsQSVM/circuit_base_corral.png')
print("Imagen del circuito base guardada como 'circuit_base.png'.")

# Usar una muestra de X_train para calcular las métricas
time_init = time.time()
metrics = calculate_gate_metrics(qc.decompose(), X_train.iloc[0].values)
time_end = time.time()

print(f"The time to compute the GSI is {time_end - time_init}")

for m in metrics:
    print(f"Puerta: {m['gate']} en posición {m['position']}")
    print(f"  Fidelidad (F): {m['F']:.4f}")
    print(f"  Entrelazamiento (E): {m['E']:.4f}")
    print(f"  (1 - Entropía) (1 - S): {1 - m['S']:.4f}")
    print(f"  (1 - Sensibilidad) (1 - P): {1 - m['P']:.4f}")
    print(f"  Métrica (M): {m['M']:.4f}\n")

# Extraer los valores de la métrica M
M_values = [m['M'] for m in metrics]

# Calcular el mínimo y máximo de M
M_min = min(M_values)
M_max = max(M_values)

print(f"Métrica (M): mínimo = {M_min:.4f}, máximo = {M_max:.4f}")

# Crear una lista de umbrales desde M_min hasta M_max en incrementos de 0.02
thresholds = np.arange(M_min, M_max + 0.02, 0.02)
thresholds = np.round(thresholds, 4)  # Redondear para evitar problemas de precisión flotante

print(thresholds)

# Diccionarios para almacenar los resultados, modelos y circuitos
results = {}
models = {}
circuits = {}

# Iterar sobre los umbrales y realizar el proceso
for idx, threshold in enumerate(thresholds):
    print(f"\nEvaluando con umbral M_threshold = {threshold}")
    try:
        # Eliminar las puertas con M < threshold
        new_feature_map = remove_low_M_gates(qc, metrics, M_threshold=threshold)

        # Verificar el número de puertas en el nuevo circuito
        numero_de_puertas = len(new_feature_map.decompose().data)
        print(
            f"El circuito tiene {numero_de_puertas} puertas después de eliminar con M_threshold = {threshold}")

        # Entrenar el modelo utilizando el feature map modificado
        model, score, total_time = pegasusQSVCTrain(
            X_train, y_train, X_val, y_val, new_feature_map)

        print(
            f"El puntaje del circuito con M_threshold = {threshold} es {score} y el tiempo es {total_time}")

        # Almacenar los resultados, el modelo y el circuito
        results[threshold] = {'Threshold': threshold, 'Score': score,
                              'Time': total_time, 'Num_Puertas': numero_de_puertas}
        models[threshold] = model
        circuits[threshold] = new_feature_map

    except Exception as e:
        print(
            f"Error al evaluar el circuito con M_threshold = {threshold}: {e}")
        # Aún así, almacenar los resultados
        results[threshold] = {'Threshold': threshold, 'Score': None,
                              'Time': None, 'Num_Puertas': numero_de_puertas}
        models[threshold] = None
        circuits[threshold] = None

# Convertir el diccionario de resultados en un DataFrame de pandas
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index.name = 'Threshold'

# Exportar el DataFrame a un archivo CSV
results_df.to_csv('resultsCorral.csv')

print("\nLos resultados han sido guardados en 'results.csv'.")

# Seleccionar el circuito base
print("\nSeleccionando el circuito base...")
base_threshold = thresholds[0]
base_result = results[base_threshold]
base_accuracy = base_result['Score']
base_time = base_result['Time']
print(f"Circuito base en threshold {base_threshold} con score {base_accuracy} y tiempo {base_time}")

# Seleccionar el mejor circuito en términos de accuracy
print("\nSeleccionando el mejor circuito en términos de accuracy...")
# Crear un DataFrame sin el modelo base
results_df_no_base = results_df.drop(index=base_threshold)

# Seleccionar el modelo con el mayor accuracy
best_accuracy_model_info = results_df_no_base.loc[results_df_no_base['Score'].idxmax()]
best_accuracy_threshold = best_accuracy_model_info['Threshold']
best_accuracy_model = models[best_accuracy_threshold]
best_accuracy_circuit = circuits[best_accuracy_threshold]
print(
    f"Mejor modelo en términos de accuracy está en threshold {best_accuracy_threshold} con score {best_accuracy_model_info['Score']} y tiempo {best_accuracy_model_info['Time']}")

# Guardar la imagen del mejor circuito en términos de accuracy
best_accuracy_circuit.draw('mpl', filename='circuitsQSVM/circuit_best_accuracy_corral.png')
print("Imagen del mejor circuito en términos de accuracy guardada como 'circuit_best_accuracy.png'.")

# Seleccionar el mejor circuito en términos de tiempo
print("\nSeleccionando el mejor circuito en términos de tiempo...")
# Definir el umbral de accuracy mínimo (opcional)
min_accuracy = base_accuracy * 0.85  # Opcional: limitar a modelos con accuracy no más de un 15% peor que el base

# Filtrar los candidatos para tiempo
candidates_time = results_df_no_base[results_df_no_base['Score'] >= min_accuracy]

if not candidates_time.empty:
    # Seleccionar el modelo con el menor tiempo
    best_time_model_info = candidates_time.loc[candidates_time['Time'].idxmin()]
    best_time_threshold = best_time_model_info['Threshold']
    best_time_model = models[best_time_threshold]
    best_time_circuit = circuits[best_time_threshold]
    print(
        f"Mejor modelo en términos de tiempo está en threshold {best_time_threshold} con score {best_time_model_info['Score']} y tiempo {best_time_model_info['Time']}")

    # Guardar la imagen del mejor circuito en términos de tiempo
    best_time_circuit.draw('mpl', filename='circuitsQSVM/circuit_best_time_corral.png')
    print("Imagen del mejor circuito en términos de tiempo guardada como 'circuit_best_time.png'.")
else:
    print("No hay modelos que cumplan con la restricción de accuracy para optimizar tiempo.")
    best_time_model = None
    best_time_circuit = None

# === Nuevo: Seleccionar el circuito basado en el ranking ===
print("\nSeleccionando el circuito basado en el ranking personalizado...")

# Calcular el ranking para cada circuito
ranking_scores = {}

for threshold, row in results_df_no_base.iterrows():
    Ai = row['Score']
    Ti = row['Time']
    if Ai is not None and Ti is not None:
        # Calcular el score de ranking según tu fórmula
        ranking_score = (base_accuracy - Ai) + ((base_time - Ti) / base_time)
        ranking_scores[threshold] = ranking_score
    else:
        # Si no hay datos de accuracy o tiempo, asignar un ranking muy bajo
        ranking_scores[threshold] = float('inf')  # Peor ranking posible

# Convertir el ranking a un DataFrame
ranking_df = pd.DataFrame.from_dict(ranking_scores, orient='index', columns=['Ranking_Score'])
ranking_df.index.name = 'Threshold'

# Combinar con los resultados originales
results_with_ranking = results_df_no_base.join(ranking_df)

print(results_with_ranking)

# Ordenar los circuitos según el ranking
results_with_ranking_sorted = results_with_ranking.sort_values(by='Ranking_Score', ascending=False)

# Seleccionar el circuito con el mejor ranking
best_ranking_threshold = results_with_ranking_sorted.index[0]
best_ranking_model_info = results_with_ranking_sorted.iloc[0]
best_ranking_model = models[best_ranking_threshold]
best_ranking_circuit = circuits[best_ranking_threshold]

print(
    f"Mejor modelo basado en ranking está en threshold {best_ranking_threshold} con ranking score {best_ranking_model_info['Ranking_Score']}, score {best_ranking_model_info['Score']} y tiempo {best_ranking_model_info['Time']}")

# Guardar la imagen del mejor circuito basado en ranking
best_ranking_circuit.draw('mpl', filename='circuitsQSVM/circuit_best_ranking_corral.png')
print("Imagen del mejor circuito basado en ranking guardada como 'circuit_best_ranking.png'.")

# === Evaluación en el conjunto de prueba ===

# Evaluar los modelos en el conjunto de prueba
print("\n=== Evaluación en el conjunto de test ===")

# Evaluar el modelo base en el conjunto de prueba
print("\nEvaluando el modelo base en el conjunto de test...")
base_model = models[base_threshold]
base_test_score, base_test_time = pegasusQSVCTest(base_model, X_test, y_test)
print(f"Accuracy del modelo base en el conjunto de test: {base_test_score}")

# Evaluar el mejor modelo en términos de accuracy en el conjunto de prueba
if best_accuracy_model is not None:
    print("\nEvaluando el mejor modelo en términos de accuracy en el conjunto de test...")
    best_accuracy_test_score, best_accuracy_test_time = pegasusQSVCTest(best_accuracy_model, X_test, y_test)
    print(f"Accuracy del mejor modelo en términos de accuracy en el conjunto de test: {best_accuracy_test_score}")
else:
    print("No hay un mejor modelo en términos de accuracy para evaluar en el conjunto de prueba.")

# Evaluar el mejor modelo en términos de tiempo en el conjunto de prueba
if best_time_model is not None:
    print("\nEvaluando el mejor modelo en términos de tiempo en el conjunto de test...")
    best_time_test_score, best_time_test_time = pegasusQSVCTest(best_time_model, X_test, y_test)
    print(f"Accuracy del mejor modelo en términos de tiempo en el conjunto de prueba: {best_time_test_score}")
else:
    print("No hay un mejor modelo en términos de tiempo para evaluar en el conjunto de test.")

# Evaluar el mejor modelo basado en ranking en el conjunto de prueba
if best_ranking_model is not None:
    print("\nEvaluando el mejor modelo basado en ranking en el conjunto de test...")
    best_ranking_test_score, best_ranking_test_time = pegasusQSVCTest(best_ranking_model, X_test, y_test)
    print(f"Accuracy del mejor modelo basado en ranking en el conjunto de prueba: {best_ranking_test_score}")
else:
    print("No hay un mejor modelo basado en ranking para evaluar en el conjunto de prueba.")
