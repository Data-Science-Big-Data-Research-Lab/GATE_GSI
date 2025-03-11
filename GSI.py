from qiskit.quantum_info import Statevector, state_fidelity, entropy, partial_trace
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import PegasosQSVC
import time
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit.quantum_info import DensityMatrix
import os

os.environ["QISKIT_AER_CPU_THREADS"] = str(os.cpu_count())
os.environ["QISKIT_AER_CUQUANTUM"] = "1"

os.environ["CUQUANTUM_MGPU"] = "1"


# Set random seed for reproducibility
np.random.seed(12345)
algorithm_globals.random_seed = 12345

def calculate_GSI_DM(feature_map, parameter_binds):
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


def calculate_GSI_MPS(feature_map, parameter_binds):
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

        # ============================================================
        # CALCULATING Entanglement (E) WITHOUT partial_trace(...) of Qiskit
        # ============================================================
        if qc.num_qubits > 1:
            # We use the MPS to obtain the submatrix of qubit 1
            rho_reduced = partial_trace_mps(current_qc, keep_qubits=[1])
            E_val = entropy(rho_reduced, base=2)  # entropy in bits
            E_normalized = E_val / 1.0  # For a qubit, the maximum entropy is 1
        else:
            E_normalized = 0.0

        # ============================================================
        # CALCULATE TOTAL ENTROPY (S)
        # ============================================================
        # If you want the entropy of the whole state, you have 2^n amplitudes.
        # You have to build its "global rho" or use the same logic (although
        # it is more expensive). For simplicity, if n is large, be careful with memory.
        # Here we can use a trick: the entropy of a pure state is 0.
        # If current_state is pure (Statevector), its total entropy S = 0.
        #
        # If you necessarily want to use the "global" entropy of a pure state,
        # it is always 0. I leave you the Qiskit code (entropy(current_state)) for
        # consistency, although you will know that it returns 0 if it is a pure statevector.

        if isinstance(current_state, Statevector):
            # Pure state by definition
            S = 0.0
        else:
            # So we assume it is a DensityMatrix
            # We verify purity
            purity = current_state.purity()
            if np.isclose(purity, 1.0, rtol=1e-12):
                S = 0.0
            else:
                # Mixed state
                S = entropy(current_state)

        # ============================================================
        # SENSITIVITY (P)
        # ============================================================
        if gate.params:
            param_values = [params[0] + delta for delta in [0, 0.01, -0.01]]
            P = calculate_sensitivity(current_qc, param_values, current_state)
        else:
            P = 0.0

        # ============================================================
        # WE COMBINE METRICS IN M
        # ============================================================
        M = (F + E_normalized + (1 - S) + (1 - P)) / 4

        # Store metrics
        metrics.append({
            'gate': gate.name,
            'position': i,
            'F': F,
            'E': E_normalized,
            'S': S,
            'P': P,
            'M': M
        })

    return metrics

def calculate_GSI_MPS_parallel(feature_map, parameter_binds):
    """
    Computes the metrics F (fidelity), E (normalized entanglement),
    S (normalized entropy), P (sensitivity), and the combined metric M (GSI)
    for each gate in the circuit. Uses parallelization to speed up
    the computation of metrics at each gate, once the states are known.
    """

    # 1) We build the circuit with the assigned parameters
    qc = feature_map.assign_parameters(parameter_binds).decompose()

    # 2) We sequentially generate the states after each gate

    current_qc = QuantumCircuit(qc.num_qubits)
    initial_state = Statevector.from_label('0' * qc.num_qubits)
    states = [initial_state]

    gate_info_list = []
    for i, instruction in enumerate(qc.data):
        gate = instruction.operation
        qubits = instruction.qubits
        qubit_indices = [q._index for q in qubits]

        current_qc.append(gate, qubit_indices)
        new_state = Statevector.from_instruction(current_qc)

        states.append(new_state)
        gate_info_list.append((i, gate, qubit_indices))

    def compute_metrics_for_gate(index):
        i, gate, qubit_indices = gate_info_list[index]
        previous_state = states[index]
        current_state = states[index + 1]

        # 3.1) Fidelity (F)
        F = state_fidelity(previous_state, current_state)

        # 3.2) Entanglement (E) normalized
        if qc.num_qubits > 1:
            rho_reduced = partial_trace_mps(current_qc, keep_qubits=[1])
            E_val = entropy(rho_reduced, base=2)  # entropy in bits
            E_normalized = E_val / 1.0
        else:
            E_normalized = 0.0

        # 3.3) Normalized total entropy (S)
        if isinstance(current_state, Statevector):
            S = 0.0
        else:
            purity = current_state.purity()
            if np.isclose(purity, 1.0, rtol=1e-12):
                S = 0.0
            else:
                S = entropy(current_state)

        # 3.4) Sensitivity (P) only if the gate has parameters
        if gate.params:
            param_values = [gate.params[0], gate.params[0] + 0.01, gate.params[0] - 0.01]
            P = calculate_sensitivity_Parallel(current_qc, param_values, current_state)
        else:
            P = 0.0

        # 3.5) Combined Metric M (GSI)
        M = (F + E_normalized + (1 - S) + (1 - P)) / 4

        return {
            'gate': gate.name,
            'position': i,
            'F': F,
            'E': E_normalized,
            'S': S,
            'P': P,
            'M': M
        }

    # 4) We parallelize the calculation of metrics at each gate
    metrics = []
    num_threads = os.cpu_count()  # Gets the number of logical threads available
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_metrics_for_gate, idx)
                   for idx in range(len(gate_info_list))]
        for f in futures:
            metrics.append(f.result())

    return metrics

def partial_trace_mps(qc, keep_qubits):
    """
    Uses the Qiskit Aer MPS simulator to obtain the reduced density matrix
    of 'keep_qubits', plotting (discarding) the remainder.
    Returns a DensityMatrix object.
    """
    # We created the simulator with MPS method
    sim = AerSimulator(method='matrix_product_state')

    # We copy the circuit and add the instruction to save the subarray,
    # specifying both parameters by name
    qc_copy = qc.copy()
    qc_copy.save_density_matrix(qubits=keep_qubits, label='rho_subsys')

    # We run the simulation
    job = sim.run(qc_copy)
    result = job.result()

    # We extract the reduced density matrix (partial trace)
    rho_data = result.data(0)['rho_subsys']
    rho_sub = DensityMatrix(rho_data)
    return rho_sub


def calculate_sensitivity_Parallel(current_qc, param_values, original_state):
    fidelities = []

    # We take the last instruction you added (the gate at position -1)
    last_instruction = current_qc.data[-1]
    last_gate = last_instruction.operation
    last_qubits = last_instruction.qubits
    last_clbits = last_instruction.clbits

    # If there are no parameters, there is nothing to disturb.
    if not last_gate.params:
        return 0.0

    GateClass = last_gate.__class__  # e.g. RXGate, RZGate, etc.
    original_params = list(last_gate.params)

    for val in param_values:
        # We create a copy of the circuit
        qc_copy = current_qc.copy()

        # We build a new gate with the same parameters except
        # the first one, which we change to 'val'
        new_params = [val] + original_params[1:]
        new_gate = GateClass(*new_params)

        # We replace the last instruction
        qc_copy.data[-1] = (new_gate, last_qubits, last_clbits)

        # We calculate the new state
        new_state = Statevector.from_instruction(qc_copy)
        fidelity = state_fidelity(original_state, new_state)
        fidelities.append(fidelity)

    # For example, we define sensitivity as the standard deviation
    P = np.std(fidelities)
    return P


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
    return ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement='linear', parameter_prefix="theta")


def remove_low_M_gates(feature_map, metrics, M_threshold=0.6):
    """
    Remove gates with M values below a given threshold from the feature map.
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

def pegasusQSVCTrainNoiseSimulator(X_train, y_train, X_val, y_val, feature_map, gpu):
    """
    Train a Pegaso QSVC using Fidelity Quantum Kernel with real ibm_brisbane noise (using AerSimulator).
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # Enable both GPUs

    # 1) Connect to IBM and pull backend "ibm_brisbane"
    # Make sure you have configured credentials with QiskitRuntimeService
    server = QiskitRuntimeService()
    real_backend = server.backend("ibm_brisbane")

    # 2) Get the actual NoiseModel from ibm_brisbane
    noise_model = NoiseModel.from_backend(real_backend)

    # 3) Configure the simulator with the 'tensor_network' method (noise supported),
    # less parallelization and more memory (for example 16 GB).
    # Adjust according to your real resources.
    sim_backend = AerSimulator(
        method="tensor_network",
        noise_model=noise_model,
        device='GPU',
        batched_shots_gpu=True,
        blocking_enable=True,
        blocking_qubits=20
    )

    # 5) Create a Sampler with the simulator configured, with noise mitigation technique (resilience level 2)
    sampler = BackendSampler(backend=sim_backend, options={"resilience_level": 2})

    # 6) Create "Compute-Uncompute Fidelity" using that sampler
    fidelity = ComputeUncompute(sampler=sampler)

    # 7) Create a FidelityQuantumKernel that uses the feature map
    qkernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity
    )

    # 8) Training the PegasosQSVC
    X_train_arr, X_val_arr = X_train.to_numpy(), X_val.to_numpy()
    y_train_arr, y_val_arr = y_train.to_numpy(), y_val.to_numpy()

    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=5000, num_steps=500)

    start_time = time.time()
    pegasos_qsvc.fit(X_train_arr, y_train_arr)
    train_time = time.time() - start_time
    print(f"PegasosQSVC training time: {train_time:.2f} s")

    # 9) Validation
    start_time = time.time()
    validation_score = pegasos_qsvc.score(X_val_arr, y_val_arr)
    validation_time = time.time() - start_time
    print(f"PegasosQSVC validation time: {validation_time:.2f} s")
    print(f"PegasosQSVC classification validation score: {validation_score:.3f}")

    return pegasos_qsvc, validation_score, (train_time + validation_time)





def pegasusQSVCTrain(X_train, y_train, X_val, y_val, feature_map):
    """
    Train a PegasosQSVC model using the Fidelity Quantum Kernel with a specified feature map.
    """

    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=5000, num_steps=500)

    # Convert training and validation data to numpy arrays
    train_features, val_features = X_train.to_numpy(), X_val.to_numpy()
    train_labels, val_labels = y_train.to_numpy(), y_val.to_numpy()

    assert feature_map.num_qubits == train_features.shape[1], \
        "Mismatch between the feature map qubits and input data features."

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