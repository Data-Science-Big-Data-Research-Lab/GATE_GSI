from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, entropy, partial_trace
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit.circuit.library import RealAmplitudes
import time

# Set random seed for reproducibility
np.random.seed(12345)
algorithm_globals.random_seed = 12345

def calculate_GSI(feature_map, parameter_binds):
    """
    Calculate F (fidelity), E (entanglement), S (entropy), P (sensitivity),
    and GSI (Gate Significance Index) for each gate in the quantum circuit.
    """
    # Assign parameter values to the feature map
    qc = feature_map.assign_parameters(parameter_binds).decompose()

    metrics = []
    current_qc = QuantumCircuit(qc.num_qubits)
    initial_state = Statevector.from_label('0' * qc.num_qubits)
    current_state = initial_state

    for i, instruction in enumerate(qc.data):
        gate = instruction.operation
        qubits = instruction.qubits
        params = gate.params

        qubit_indices = [qubit._index for qubit in qubits]

        # Save the previous state
        previous_state = current_state.copy()

        # Add the gate to the current circuit
        current_qc.append(gate, qubit_indices)

        # Get the new state
        current_state = Statevector.from_instruction(current_qc)

        # Calculate fidelity
        F = state_fidelity(previous_state, current_state)

        # Calculate entanglement (normalized)
        if qc.num_qubits > 1:
            reduced_state = partial_trace(current_state, [1])
            E = entropy(reduced_state)
            max_entropy = np.log2(reduced_state.dim)
            E_normalized = E / max_entropy if max_entropy != 0 else 0
        else:
            E_normalized = 0.0

        # Calculate total entropy (normalized)
        S = entropy(current_state)
        max_entropy_total = np.log2(current_state.dim)
        S_normalized = S / max_entropy_total if max_entropy_total != 0 else 0

        # Calculate sensitivity for parameterized gates
        if gate.params:
            param_values = [params[0] + delta for delta in [0, 0.01, -0.01]]
            P = calculate_sensitivity(current_qc, param_values, current_state)
        else:
            P = 0.0

        # Combine metrics into M
        M = (F + E_normalized + (1 - S_normalized) + (1 - P)) / 4

        # Store metrics
        metrics.append({
            'gate': gate.name,
            'position': i,
            'F': F,
            'E': E_normalized,
            'S': S_normalized,
            'P': P,
            'M': M
        })

    return metrics


def calculate_sensitivity(current_qc, param_values, original_state):
    """
       Calculate sensitivity (P) of a parameterized gate by slightly varying its parameter
       and measuring the resulting state fidelity.
       """
    fidelities = []
    for val in param_values:
        qc_copy = current_qc.copy()
        qc_copy.data[-1].operation.params[0] = val
        new_state = Statevector.from_instruction(qc_copy)
        fidelity = state_fidelity(original_state, new_state)
        fidelities.append(fidelity)
    P = np.std(fidelities)
    return P


def create_feature_map(num_qubits):
    """
    Create a ZZFeatureMap with a specified number of qubits and linear entanglement.
    """
    return ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement='linear')


def remove_low_M_gates(feature_map, metrics, M_threshold=0.6):
    """
    Remove gates with GSI values below a given threshold from the feature map.
    """
    feature_map_decomposed = feature_map.decompose()
    new_circuit = QuantumCircuit(feature_map.num_qubits)

    # Identify gates to keep based on M threshold
    positions_to_keep = sorted([m['position'] for m in metrics if m['M'] >= M_threshold])

    # Rebuild the circuit with selected gates
    for idx, instruction in enumerate(feature_map_decomposed.data):
        if idx in positions_to_keep:
            new_circuit.append(instruction.operation, instruction.qubits)

    return new_circuit


def pegasusQSVCTrain(X_train, y_train, X_val, y_val, feature_map):
    """
    Train a PegasosQSVC model using the Fidelity Quantum Kernel with a specified feature map.
    """
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=5000, num_steps=500)

    # Convert training and validation data to numpy arrays
    train_features, val_features = X_train.to_numpy(), X_val.to_numpy()
    train_labels, val_labels = y_train.to_numpy(), y_val.to_numpy()

    # Train the model
    start_time = time.time()
    pegasos_qsvc.fit(train_features, train_labels)
    training_time = time.time() - start_time

    print(f'PegasosQSVC training time: {training_time}')

    # Validate the model
    start_time = time.time()
    validation_score = pegasos_qsvc.score(val_features, val_labels)
    validation_time = time.time() - start_time

    print(f'PegasosQSVC validation time: {validation_time}')
    print(f"PegasosQSVC classification validation score: {validation_score}")

    return pegasos_qsvc, validation_score, training_time + validation_time


def pegasusQSVCTest(model, X_test, y_test):
    """
    Test the trained PegasosQSVC model on a test dataset.
    """
    # Convert the test dataset to numpy arrays
    test_features = X_test.to_numpy()
    test_labels = y_test.to_numpy()

    # Measure test performance
    start_time = time.time()
    pegasos_score = model.score(test_features, test_labels)
    testing_time = time.time() - start_time

    print(f'PegasosQSVC testing time: {testing_time}')
    print(f"PegasosQSVC classification test score: {pegasos_score}")

    return pegasos_score, testing_time
