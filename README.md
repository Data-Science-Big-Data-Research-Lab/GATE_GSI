# GATE (Gate Assessment and Threshold Evaluation) and GSI (Gate Significance Index)

## Overview

This repository introduces a novel methodology (GATE) for optimizing quantum circuits in quantum machine learning (QML). The optimization is based on a newly developed **Gate Significance Index (GSI)**, which quantifies the contribution of individual quantum gates within a circuit. By identifying and removing gates with minimal computational impact, this approach enhances the performance of quantum circuits, reducing execution time and improving computational accuracy.

GATE has been validated using real-world datasets, demonstrating significant improvements in quantum circuit efficiency and robustness. This work contributes to quantum computing by providing a systematic, approach to optimizing quantum circuits. To reproduce the results, it's important to download the qiskit 1.2.4 and qiskit_machine_learning 0.7.2 versions.

## Key Features

- **Gate Significance Index (GSI):** A novel metric for evaluating the significance of each gate in a quantum circuit.
- **Optimization:** Systematic gate removal to reduce circuit complexity, minimize errors, and enhance execution time.
- **Accuracy Improvement:** Reducing unnecessary gates often increases computational accuracy by mitigating noise effects.
- **Scalability:** Compatible with various datasets and quantum algorithms.
- **Extensibility:** Open-source framework for researchers to adapt and extend the methodology.

## Gate Significance Index (GSI)

The **GSI** is a metric designed to evaluate the significance of each quantum gate in a circuit. It is based on the following metrics:

1. **Fidelity (F)**: Assesses state change accuracy.
2. **Entanglement (E)**: Measures quantum correlations created by the gate.
3. **Entropy (S)**: Indicates the disorder introduced by the gate.
4. **Sensitivity (P)**: Captures the gate's reaction to parameter variations.

The **GSI formula** combines these metrics:

$GSI = \frac{F + E + (1 - S) + (1 - P)}{4}, GSI \in [0, 1] $

## Gate Assessment and Threshold Evaluation (GATE)

The **GATE** methodology applies GSI to optimize quantum circuits through these steps:

1. **Data Preparation:**
   - Split datasets into training (60%), validation (20%), and test (20%).
   - Use angle encoding to map classical data to quantum states.

2. **GSI Calculation:**
   - Compute GSI for each gate using the metrics defined above.

3. **Threshold Selection:**
   - Set a GSI threshold. Gates below this threshold are marked for removal.

4. **Gate Removal:**
   - Iteratively prune low-significance gates to simplify the circuit.

5. **QML Model Generation:**
   - Use the optimized circuit to create a new quantum machine learning (QML) model, such as PegasosQSVM.

6. **Performance Assessment:**
   - Evaluate models on accuracy and execution time using the validation set.
   
7. **Iterative Refinement:**
   - Increase the GSI threshold incrementally to explore different quantum circuits configurations.
     
8. **Model Ranking:**
   - Rank models by:
     - Accuracy. Best Accuracy
     - Time. Fastest execution (ensures at least 85% of baseline accuracy).
     - Balance. Trade-off between accuracy and execution time.
     
9. **Final Testing:**
    - Validate the top-ranked models to ensure reliability.
      
## Results

- **Datasets:** Tested on six datasets, including BreastW, Fitness, Glass2, Heart, Corral, and Diabetes.
- **Performance Gains:**
  - Gate reduction: Up to **40% fewer gates**.
  - Execution time improvement: Up to **50% faster**.
  - Accuracy improvement: Up to **12.63% higher accuracy** in some cases.
- **Adaptability:** Demonstrated effectiveness across diverse datasets and problem domains.

## Contact

For questions, feedback, or further information, please contact:

- **F. Rodríguez-Díaz:** [froddia@upo.es](mailto:froddia@upo.es)


