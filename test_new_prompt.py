#!/usr/bin/env python3
"""
Test script to validate the updated prompt.md format for common mistakes.
This script simulates what the model should produce based on our new requirements.
"""

import json
from datetime import datetime

def create_test_dataset():
    """Create a test dataset following the new common mistakes format."""
    
    test_dataset = {
        "problem_id": "test_grover_2qubit",
        "problem_description": "Test implementation of Grover's algorithm for 2-qubit system targeting |11⟩ state.",
        "difficulty": "intermediate",
        "category": "quantum_algorithms",
        "learning_objectives": ["Grover's algorithm implementation", "Oracle construction"],
        "prerequisites": ["Basic quantum gates", "Quantum superposition"],
        
        "reasoning_trace": {
            "problem_analysis": "For 2-qubit system, need to search for |11⟩ state with optimal iterations.",
            "quantum_physics_reasoning": "Grover's algorithm amplifies target state amplitude through oracle and diffusion.",
            "algorithm_choice": "Standard Grover's algorithm optimal for unstructured search.",
            "parameter_selection": "Optimal iterations = floor(π/4 * sqrt(N/M)) ≈ 1 for N=4, M=1",
            "implementation_strategy": "Initialize superposition, apply oracle+diffusion, measure."
        },
        
        "solutions": [
            {
                "preference_rank": 1,
                "code": """from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def create_grover_2qubit():
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialize superposition
    qc.h(qr)
    
    # Oracle for |11⟩
    qc.cz(qr[0], qr[1])
    
    # Diffusion operator
    qc.h(qr)
    qc.x(qr)
    qc.cz(qr[0], qr[1])
    qc.x(qr)
    qc.h(qr)
    
    # Measure
    qc.measure(qr, cr)
    return qc

# Execute
circuit = create_grover_2qubit()
simulator = AerSimulator()
transpiled_circuit = transpile(circuit, simulator)
job = simulator.run(transpiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)""",
                "code_reasoning": {
                    "structure_choice": "Modular function design for clarity",
                    "gate_implementation": "CZ gate for oracle, standard diffusion pattern",
                    "optimization_decisions": "Minimal gate count for 2-qubit system",
                    "measurement_strategy": "Direct measurement of both qubits"
                },
                "expected_output": "{'11': ~1000}",
                "output_interpretation": "High probability of |11⟩ confirms successful Grover's implementation"
            }
        ],
        
        "common_mistakes": [
            {
                "incorrect_code": """from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def create_grover_2qubit():
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialize superposition
    qc.h(qr)
    
    # WRONG: Oracle marks |00⟩ instead of |11⟩
    qc.x(qr)  # Flip to make |11⟩ -> |00⟩
    qc.cz(qr[0], qr[1])  # This now marks |00⟩
    qc.x(qr)  # Flip back
    
    # Diffusion operator  
    qc.h(qr)
    qc.x(qr)
    qc.cz(qr[0], qr[1])
    qc.x(qr)
    qc.h(qr)
    
    # Measure
    qc.measure(qr, cr)
    return qc

# Execute
circuit = create_grover_2qubit()
simulator = AerSimulator()
transpiled_circuit = transpile(circuit, simulator)
job = simulator.run(transpiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)""",
                "error_type": "conceptual_misunderstanding",
                "explanation": "The oracle marks |00⟩ instead of the intended |11⟩ state due to incorrect X gate placement.",
                "debugging_reasoning": "Check which state the oracle actually marks by tracing through the gate operations. The X gates transform |11⟩ to |00⟩ before applying CZ.",
                "expected_output": "{'00': ~1000}"
            },
            {
                "incorrect_code": """from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def create_grover_2qubit():
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialize superposition
    qc.h(qr)
    
    # Oracle for |11⟩
    qc.cz(qr[0], qr[1])
    
    # WRONG: Missing diffusion operator - algorithm won't work!
    # qc.h(qr)
    # qc.x(qr) 
    # qc.cz(qr[0], qr[1])
    # qc.x(qr)
    # qc.h(qr)
    
    # Measure
    qc.measure(qr, cr)
    return qc

# Execute
circuit = create_grover_2qubit()
simulator = AerSimulator()
transpiled_circuit = transpile(circuit, simulator)
job = simulator.run(transpiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)""",
                "error_type": "bug",
                "explanation": "Missing diffusion operator means the algorithm doesn't amplify the marked state. Oracle alone just adds a phase.",
                "debugging_reasoning": "Grover's algorithm requires both oracle AND diffusion operator. Without diffusion, there's no amplitude amplification.",
                "expected_output": "{'00': ~250, '01': ~250, '10': ~250, '11': ~250}"
            }
        ],
        
        "validation_tests": [
            {
                "test_description": "Verify |11⟩ has high probability",
                "test_code": "assert counts.get('11', 0) > 800, f'Expected high |11⟩ count, got {counts}'",
                "expected_test_result": "Should pass for correct implementation"
            }
        ],
        
        "extensions": [
            "Extend to 3-qubit system",
            "Multiple marked states",
            "Variable iteration counts"
        ]
    }
    
    return test_dataset

def test_validation_compatibility():
    """Test that the new format works with our validation system."""
    print("Testing new common mistakes format compatibility...")
    
    dataset = create_test_dataset()
    
    # Check that common mistakes have complete code
    for i, mistake in enumerate(dataset["common_mistakes"]):
        code = mistake["incorrect_code"]
        print(f"\nCommon Mistake {i+1}:")
        print(f"  Lines of code: {len(code.strip().split())}")
        print(f"  Has imports: {'from qiskit import' in code}")
        print(f"  Has execution: {'simulator.run' in code}")
        print(f"  Expected output: {mistake['expected_output']}")
        
        # Verify it's complete, executable code
        assert "from qiskit import" in code, "Missing imports"
        assert "QuantumCircuit" in code, "Missing circuit creation"
        assert "simulator.run" in code, "Missing execution"
        assert "expected_output" in mistake, "Missing expected output"
    
    print("\n✅ All common mistakes have complete, executable code!")
    
    # Save test dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_dataset_new_format_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Test dataset saved to: {filename}")
    return filename

if __name__ == "__main__":
    test_file = test_validation_compatibility()
    print(f"\nYou can now test validation with: python validate_dataset.py {test_file}")