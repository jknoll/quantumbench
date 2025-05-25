#!/usr/bin/env python3
"""
Generated Python code for: bell_state_preparation_parameterized
Extracted from: 2.json

This file has been fixed to be executable standalone with default parameters.
You can modify the params dictionary below to test different parameter values.
"""

params = {
    'bell_type': 'phi_plus',  # Type of Bell state to prepare
}
shots = 1000

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

bell_type = params['bell_type']

qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qr, cr)

circuit.h(qr[0])

if bell_type == 'phi_minus':
    circuit.z(qr[0])
elif bell_type == 'psi_minus':
    circuit.z(qr[0])

circuit.cx(qr[0], qr[1])

if bell_type in ['psi_plus', 'psi_minus']:
    circuit.x(qr[1])

circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

print("Execution completed successfully!")
print(f"Results: {counts}")
