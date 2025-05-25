#!/usr/bin/env python3
"""
Generated Python code for: grover_2qubit_parameterized
Extracted from: 1.json

This file has been fixed to be executable standalone with default parameters.
You can modify the params dictionary below to test different parameter values.
"""

params = {
    'marked_state': 0,  # Index of the computational basis state to be marked (0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩)
}
shots = 1000

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

marked_state = params['marked_state']

qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qr, cr)

# Initialize superposition
circuit.h(qr)

# Oracle for marked state
binary = format(marked_state, '02b')
for i, bit in enumerate(binary):
    if bit == '0':
        circuit.x(qr[i])

circuit.cz(qr[0], qr[1])

for i, bit in enumerate(binary):
    if bit == '0':
        circuit.x(qr[i])

# Diffusion operator
circuit.h(qr)
circuit.x(qr)
circuit.cz(qr[0], qr[1])
circuit.x(qr)
circuit.h(qr)

circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

print("Execution completed successfully!")
print(f"Results: {counts}")
