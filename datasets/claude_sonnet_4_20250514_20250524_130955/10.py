#!/usr/bin/env python3
"""
Generated Python code for: quantum_circuit_example_10
Extracted from: 10.json

This file has been fixed to be executable standalone with default parameters.
You can modify the params dictionary below to test different parameter values.
"""

params = {
    'n_qubits': 2,  # Number of qubits in the circuit
    'angle': 1.57,  # Rotation angle parameter
}
shots = 1000

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_qubits = params['n_qubits']
angle = params['angle']

qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Example 10 circuit pattern
circuit.h(qr[0])
for j in range(n_qubits - 1):
    circuit.cx(qr[j], qr[j + 1])
    circuit.ry(angle / (j + 1), qr[j])

circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

print("Execution completed successfully!")
print(f"Results: {counts}")
