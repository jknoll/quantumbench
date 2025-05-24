#!/usr/bin/env python3
"""
Create a test dataset from the raw response for validation testing.
"""

import json

# Create a simplified but valid dataset for testing
test_dataset = {
    "problem_id": "grover_2qubit_marked_state",
    "problem_description": "Implement Grover's search algorithm for a 2-qubit system to find a marked state |11⟩.",
    "difficulty": "intermediate",
    "category": "quantum_algorithms",
    "learning_objectives": [
        "Understanding Grover's algorithm components",
        "Oracle construction for specific marked states",
        "Amplitude amplification mechanics"
    ],
    "prerequisites": [
        "Basic quantum gates",
        "Quantum superposition",
        "Basic Qiskit syntax"
    ],
    "solutions": [
        {
            "preference_rank": 1,
            "code": """
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

def create_grover_circuit():
    # Initialize circuit
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Initial Hadamard gates
    circuit.h(qr)
    
    # Oracle for |11⟩
    circuit.x(qr[0])
    circuit.x(qr[1])
    circuit.h(qr[1])
    circuit.cx(qr[0], qr[1])
    circuit.h(qr[1])
    circuit.x(qr[0])
    circuit.x(qr[1])
    
    # Diffusion operator
    circuit.h(qr)
    circuit.x(qr)
    circuit.h(qr[1])
    circuit.cx(qr[0], qr[1])
    circuit.h(qr[1])
    circuit.x(qr)
    circuit.h(qr)
    
    # Measurement
    circuit.measure(qr, cr)
    
    return circuit

# Execute circuit
circuit = create_grover_circuit()
backend = Aer.get_backend('qasm_simulator')
job = backend.run(circuit, shots=1000)
result = job.result()
counts = result.get_counts()
""",
            "expected_output": "{'11': 1000}",
            "output_interpretation": "High frequency of |11⟩ measurements indicates successful search."
        }
    ],
    "common_mistakes": [
        {
            "incorrect_code": """
# Incorrect oracle implementation
circuit.x(qr)
circuit.cz(qr[0], qr[1])
circuit.x(qr)
""",
            "error_type": "conceptual_misunderstanding",
            "explanation": "This oracle only applies phase to |00⟩ instead of |11⟩."
        }
    ]
}

# Save the test dataset
with open('test_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(test_dataset, f, indent=2, ensure_ascii=False)

print("✓ Created test_dataset.json for validation testing")