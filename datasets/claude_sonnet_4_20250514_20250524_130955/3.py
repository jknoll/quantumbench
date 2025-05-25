#!/usr/bin/env python3
"""
Generated Python code for: qft_parameterized
Extracted from: 3.json

This file has been fixed to be executable standalone with default parameters.
You can modify the params dictionary below to test different parameter values.
"""

params = {
    'n_qubits': 2,  # Number of qubits in the QFT circuit
    'input_state': 0,  # Input computational basis state (must be < 2^n_qubits)
}
shots = 1000

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_qubits = params['n_qubits']
input_state = params['input_state'] % (2**n_qubits)

qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Prepare input state
binary = format(input_state, f'0{n_qubits}b')
for i, bit in enumerate(binary):
    if bit == '1':
        circuit.x(qr[i])

# QFT
for i in range(n_qubits):
    circuit.h(qr[i])
    for j in range(i + 1, n_qubits):
        angle = np.pi / (2 ** (j - i))
        circuit.cp(angle, qr[j], qr[i])

# Swap qubits
for i in range(n_qubits // 2):
    circuit.swap(qr[i], qr[n_qubits - 1 - i])

circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

print("Execution completed successfully!")
print(f"Results: {counts}")
