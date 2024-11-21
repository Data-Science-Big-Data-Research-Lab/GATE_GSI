# Quantum Circuit Optimization with Gate Significance Index (GSI)

## Overview

This repository introduces a novel methodology for optimizing quantum circuits in quantum machine learning (QML). The optimization is based on a newly developed **Gate Significance Index (GSI)**, which quantifies the contribution of individual quantum gates within a circuit. By identifying and removing gates with minimal computational impact, this approach enhances the performance of quantum circuits, reducing execution time and improving computational accuracy.

Our methodology has been validated using real-world datasets, demonstrating significant improvements in quantum circuit efficiency and robustness. This work contributes to quantum computing by providing a systematic, hardware-agnostic approach to optimizing quantum circuits, paving the way for better utilization of current and future quantum hardware.

## Key Features

- **Gate Significance Index (GSI):** A novel metric for evaluating the significance of each gate in a quantum circuit.
- **Optimization:** Systematic gate removal to reduce circuit complexity, minimize errors, and enhance execution time.
- **Accuracy Improvement:** Reducing unnecessary gates often increases computational accuracy by mitigating noise effects.
- **Scalability:** Compatible with various datasets, quantum algorithms, and hardware architectures.
- **Extensibility:** Open-source framework for researchers to adapt and extend the methodology.

## Methodology

The optimization process follows a structured pipeline:

1. **Data Preparation:**
   - Input dataset is split into training, validation, and test sets.
   - Feature mapping encodes classical data into quantum states using angle encoding.

2. **Gate Significance Index (GSI) Calculation:**
   - GSI is computed based on four key metrics:
     - **Fidelity:** Measures state similarity.
     - **Entanglement:** Quantifies quantum correlations.
     - **Entropy:** Assesses state purity.
     - **Sensitivity:** Evaluates gate impact on computational stability.

3. **Threshold Selection:**
   - Gates with GSI below a dynamic threshold are identified for removal.

4. **Gate Removal:**
   - Low-significance gates are pruned, and circuits are restructured for efficiency.

5. **Performance Assessment:**
   - Optimized circuits are evaluated on validation sets for accuracy and execution time.
   - Best models are ranked based on accuracy, time, and a balance of both.

6. **Testing:**
   - Final optimized models are tested for generalization and performance.

## Results

- **Datasets:** Tested on six datasets, including BreastW, Fitness, Glass2, Heart, corral, and Diabetes.
- **Performance Gains:**
  - Gate reduction: Up to **40% fewer gates**.
  - Execution time improvement: Up to **50% faster**.
  - Accuracy improvement: Up to **12.63% higher accuracy** in some cases.
- **Adaptability:** Demonstrated effectiveness across diverse datasets and problem domains.

## Contact

For questions, feedback, or further information, please contact:

- **F. Rodríguez-Díaz:** [froddia@upo.es](mailto:froddia@upo.es)


