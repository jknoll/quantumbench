#!/usr/bin/env python3
"""
Generated Python code for: phase_estimation_parameterized
Extracted from: 4.json

This file has been fixed to be executable standalone with default parameters.
You can modify the params dictionary below to test different parameter values.
"""

params = {
    'n_ancilla': 3,  # Number of ancilla qubits for phase estimation
    'target_phase': 0.125,  # True phase to estimate (as fraction of 2π)
}
shots = 1000

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_ancilla = params['n_ancilla']
target_phase = params['target_phase']

ancilla = QuantumRegister(n_ancilla, 'a')
target = QuantumRegister(1, 't')
cr = ClassicalRegister(n_ancilla, 'c')
circuit = QuantumCircuit(ancilla, target, cr)

# Initialize target in eigenstate
circuit.x(target[0])

# Initialize ancilla in superposition
circuit.h(ancilla)

# Controlled unitaries
for i in range(n_ancilla):
    repetitions = 2 ** i
    angle = 2 * np.pi * target_phase * repetitions
    circuit.cp(angle, ancilla[i], target[0])

# Inverse QFT on ancilla
for i in range(n_ancilla // 2):
    circuit.swap(ancilla[i], ancilla[n_ancilla - 1 - i])

for i in range(n_ancilla):
    for j in range(i):
        angle = -np.pi / (2 ** (i - j))
        circuit.cp(angle, ancilla[j], ancilla[i])
    circuit.h(ancilla[i])

circuit.measure(ancilla, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

print("Execution completed successfully!")
print(f"Results: {counts}")
