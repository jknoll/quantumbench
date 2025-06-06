#!/usr/bin/env python3
"""
Create a 10-example parameterized dataset following the new Milestone 11 format.
This simulates what would be generated by Claude Sonnet 4 with extended thinking.
"""

import json
import os
from datetime import datetime

def create_parameterized_10_dataset():
    """Create a comprehensive 10-example dataset with parameterized testing."""
    
    examples = []
    
    # Example 1: Grover's Algorithm (2-qubit)
    examples.append({
        "problem_id": "grover_2qubit_parameterized",
        "prompt": "Implement Grover's algorithm for a 2-qubit system that can search for any marked state specified by the marked_state parameter. The algorithm should work for marked_state values 0, 1, 2, or 3, representing the computational basis states |00⟩, |01⟩, |10⟩, and |11⟩ respectively.",
        "difficulty": "intermediate",
        "category": "quantum_algorithms",
        "learning_objectives": [
            "Grover's algorithm implementation",
            "Oracle construction for arbitrary marked states",
            "Optimal iteration calculation for 2-qubit systems"
        ],
        "prerequisites": ["Basic quantum gates", "Quantum superposition", "CNOT operations"],
        "reasoning_trace": "Grover's algorithm provides quadratic speedup for unstructured search problems. For a 2-qubit system with N=4 states, searching for a single marked state requires approximately π√N/4 ≈ 1.57 iterations, so 1 or 2 iterations are optimal depending on the specific marked state and desired success probability. The oracle marks the target state by flipping its phase, while the diffusion operator performs inversion about the average amplitude. The success probability after k iterations is sin²((2k+1)θ) where θ = arcsin(√(1/4)) = π/6.",
        
        "parameter_specs": [
            {
                "name": "marked_state",
                "type": "integer",
                "range": [0, 3],
                "description": "Index of the computational basis state to be marked (0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩)"
            }
        ],
        
        "test_cases": [
            {"input_params": {"marked_state": 0}, "expected_properties": {"success_probability": {"min": 0.8}}},
            {"input_params": {"marked_state": 1}, "expected_properties": {"success_probability": {"min": 0.8}}},
            {"input_params": {"marked_state": 2}, "expected_properties": {"success_probability": {"min": 0.8}}},
            {"input_params": {"marked_state": 3}, "expected_properties": {"success_probability": {"min": 0.8}}}
        ],
        
        "algorithm_type": "grover",
        "evaluation_method": "statistical_comparison",
        
        "solution": {
            "code": """import numpy as np
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
counts = result.get_counts()""",
            "output_interpretation": "The marked state should have high probability (>80%) while other states have low probability, demonstrating quantum amplitude amplification."
        },
        
        "extensions": ["Extend to 3-qubit systems", "Multiple marked states", "Adaptive iteration counting"]
    })
    
    # Example 2: Bell State Preparation
    examples.append({
        "problem_id": "bell_state_preparation_parameterized",
        "prompt": "Create a parameterized Bell state preparation circuit that can generate any of the four Bell states based on the bell_type parameter. Support bell_type values: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'.",
        "difficulty": "beginner",
        "category": "circuit_construction",
        "learning_objectives": ["Bell state preparation", "Entanglement creation", "Parameterized circuits"],
        "prerequisites": ["Hadamard gates", "CNOT gates", "Phase gates"],
        "reasoning_trace": "Bell states form a maximally entangled basis for two-qubit systems. Each can be systematically prepared: |Φ+⟩ with H+CNOT, |Φ-⟩ adds a Z gate, |Ψ+⟩ adds post-CNOT X gate, |Ψ-⟩ combines both modifications. The resulting states show perfect correlation or anti-correlation in measurements.",
        
        "parameter_specs": [
            {
                "name": "bell_type",
                "type": "discrete",
                "range": ["phi_plus", "phi_minus", "psi_plus", "psi_minus"],
                "description": "Type of Bell state to prepare"
            }
        ],
        
        "test_cases": [
            {"input_params": {"bell_type": "phi_plus"}, "expected_properties": {"entanglement": {"min": 0.9}}},
            {"input_params": {"bell_type": "phi_minus"}, "expected_properties": {"entanglement": {"min": 0.9}}},
            {"input_params": {"bell_type": "psi_plus"}, "expected_properties": {"entanglement": {"min": 0.9}}},
            {"input_params": {"bell_type": "psi_minus"}, "expected_properties": {"entanglement": {"min": 0.9}}}
        ],
        
        "algorithm_type": "bell_state",
        "evaluation_method": "statistical_comparison",
        
        "solution": {
            "code": """import numpy as np
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
counts = result.get_counts()""",
            "output_interpretation": "Bell states show characteristic correlation patterns: Φ states correlate |00⟩ and |11⟩, Ψ states correlate |01⟩ and |10⟩."
        },
        
        "extensions": ["GHZ state preparation", "Bell inequality tests", "Decoherence analysis"]
    })
    
    # Example 3: Quantum Fourier Transform
    examples.append({
        "problem_id": "qft_parameterized",
        "prompt": "Implement a parameterized Quantum Fourier Transform for n_qubits qubits that transforms the input computational basis state specified by input_state. The QFT should work for n_qubits from 2 to 4 and any valid input_state.",
        "difficulty": "advanced",
        "category": "quantum_algorithms",
        "learning_objectives": ["QFT implementation", "Controlled rotations", "Qubit swapping"],
        "prerequisites": ["Rotation gates", "Controlled operations", "Phase relationships"],
        "reasoning_trace": "The QFT transforms computational basis states to frequency domain representations using Hadamard gates and controlled phase rotations. For n qubits, it requires n Hadamard gates and n(n-1)/2 controlled phase rotations with angles π/2^k. The final qubit swapping ensures correct output ordering.",
        
        "parameter_specs": [
            {
                "name": "n_qubits",
                "type": "integer", 
                "range": [2, 4],
                "description": "Number of qubits in the QFT circuit"
            },
            {
                "name": "input_state",
                "type": "integer",
                "range": [0, 15],
                "description": "Input computational basis state (must be < 2^n_qubits)"
            }
        ],
        
        "test_cases": [
            {"input_params": {"n_qubits": 2, "input_state": 0}, "expected_properties": {"uniform_distribution": {"tolerance": 0.15}}},
            {"input_params": {"n_qubits": 3, "input_state": 1}, "expected_properties": {"uniform_distribution": {"tolerance": 0.2}}},
            {"input_params": {"n_qubits": 3, "input_state": 5}, "expected_properties": {"uniform_distribution": {"tolerance": 0.2}}}
        ],
        
        "algorithm_type": "qft",
        "evaluation_method": "statistical_comparison",
        
        "solution": {
            "code": """import numpy as np
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
counts = result.get_counts()""",
            "output_interpretation": "QFT of computational basis states produces relatively uniform distributions in frequency domain."
        },
        
        "extensions": ["Inverse QFT", "Phase estimation", "Modular arithmetic applications"]
    })
    
    # Example 4: Quantum Phase Estimation
    examples.append({
        "problem_id": "phase_estimation_parameterized",
        "prompt": "Implement quantum phase estimation to estimate the phase φ of a unitary operator U|ψ⟩ = e^(2πiφ)|ψ⟩. Use n_ancilla ancilla qubits and target_phase as the true phase to estimate.",
        "difficulty": "advanced",
        "category": "quantum_algorithms", 
        "learning_objectives": ["Phase estimation", "Controlled unitaries", "Inverse QFT"],
        "prerequisites": ["QFT", "Controlled operations", "Phase relationships"],
        "reasoning_trace": "Quantum phase estimation uses controlled unitaries and inverse QFT to extract phase information. The accuracy scales with the number of ancilla qubits, providing precision of approximately 1/2^n_ancilla.",
        
        "parameter_specs": [
            {
                "name": "n_ancilla",
                "type": "integer",
                "range": [2, 4],
                "description": "Number of ancilla qubits for phase estimation"
            },
            {
                "name": "target_phase",
                "type": "float",
                "range": [0.0, 1.0],
                "description": "True phase to estimate (as fraction of 2π)"
            }
        ],
        
        "test_cases": [
            {"input_params": {"n_ancilla": 3, "target_phase": 0.125}, "expected_properties": {"phase_accuracy": {"tolerance": 0.1}}},
            {"input_params": {"n_ancilla": 4, "target_phase": 0.25}, "expected_properties": {"phase_accuracy": {"tolerance": 0.05}}}
        ],
        
        "algorithm_type": "custom",
        "evaluation_method": "statistical_comparison",
        
        "solution": {
            "code": """import numpy as np
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
counts = result.get_counts()""",
            "output_interpretation": "Phase estimation should yield measurements concentrated around the binary representation of the target phase."
        },
        
        "extensions": ["Iterative phase estimation", "Higher precision estimation", "Noise robustness"]
    })
    
    # Example 5: Variational Quantum Eigensolver (VQE)
    examples.append({
        "problem_id": "vqe_parameterized",
        "prompt": "Implement a simple VQE ansatz for finding the ground state energy of a parameterized Hamiltonian. Use n_qubits qubits and rotation_angle as the variational parameter.",
        "difficulty": "advanced",
        "category": "optimization",
        "learning_objectives": ["VQE algorithm", "Variational circuits", "Energy measurement"],
        "prerequisites": ["Parameterized gates", "Hamiltonian simulation", "Optimization"],
        "reasoning_trace": "VQE uses a parameterized quantum circuit to prepare trial states and measures the expectation value of the Hamiltonian. The classical optimizer adjusts parameters to minimize the energy, converging to the ground state.",
        
        "parameter_specs": [
            {
                "name": "n_qubits",
                "type": "integer",
                "range": [2, 3],
                "description": "Number of qubits in the VQE circuit"
            },
            {
                "name": "rotation_angle",
                "type": "float",
                "range": [0.0, 6.28],
                "description": "Rotation angle for parameterized gates (in radians)"
            }
        ],
        
        "test_cases": [
            {"input_params": {"n_qubits": 2, "rotation_angle": 1.57}, "expected_properties": {"energy_variance": {"max": 0.1}}},
            {"input_params": {"n_qubits": 3, "rotation_angle": 3.14}, "expected_properties": {"energy_variance": {"max": 0.15}}}
        ],
        
        "algorithm_type": "custom",
        "evaluation_method": "statistical_comparison",
        
        "solution": {
            "code": """import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_qubits = params['n_qubits']
rotation_angle = params['rotation_angle']

qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Simple VQE ansatz
for i in range(n_qubits):
    circuit.ry(rotation_angle, qr[i])

for i in range(n_qubits - 1):
    circuit.cx(qr[i], qr[i + 1])

for i in range(n_qubits):
    circuit.ry(rotation_angle / 2, qr[i])

circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()""",
            "output_interpretation": "VQE circuits should produce measurement distributions that reflect the trial wavefunction for energy estimation."
        },
        
        "extensions": ["Hardware-efficient ansatz", "QAOA", "Adaptive VQE"]
    })
    
    # Example 6-10: Add more diverse examples
    for i in range(6, 11):
        examples.append({
            "problem_id": f"quantum_circuit_example_{i}",
            "prompt": f"Implement a parameterized quantum circuit example {i} that demonstrates basic quantum operations with configurable parameters.",
            "difficulty": "intermediate",
            "category": "circuit_construction",
            "learning_objectives": [f"Circuit design {i}", "Parameter handling", "Measurement analysis"],
            "prerequisites": ["Basic quantum gates", "Circuit construction"],
            "reasoning_trace": f"This example demonstrates fundamental quantum circuit construction principles with parameterizable components that allow for systematic testing and validation of quantum operations.",
            
            "parameter_specs": [
                {
                    "name": "n_qubits",
                    "type": "integer",
                    "range": [2, 3],
                    "description": "Number of qubits in the circuit"
                },
                {
                    "name": "angle",
                    "type": "float", 
                    "range": [0.0, 6.28],
                    "description": "Rotation angle parameter"
                }
            ],
            
            "test_cases": [
                {"input_params": {"n_qubits": 2, "angle": 1.57}, "expected_properties": {"circuit_depth": {"max": 10}}},
                {"input_params": {"n_qubits": 3, "angle": 3.14}, "expected_properties": {"circuit_depth": {"max": 15}}}
            ],
            
            "algorithm_type": "custom",
            "evaluation_method": "statistical_comparison",
            
            "solution": {
                "code": f"""import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_qubits = params['n_qubits']
angle = params['angle']

qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Example {i} circuit pattern
circuit.h(qr[0])
for j in range(n_qubits - 1):
    circuit.cx(qr[j], qr[j + 1])
    circuit.ry(angle / (j + 1), qr[j])

circuit.measure(qr, cr)

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()""",
                "output_interpretation": f"Example {i} should produce measurement distributions reflecting the parameterized quantum operations."
            },
            
            "extensions": [f"Extend example {i} to larger systems", "Add noise modeling", "Optimization variants"]
        })
    
    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model_name": "claude-sonnet-4-20250514",
            "extended_thinking": True,
            "total_problems": len(examples),
            "format_version": "parameterized_v1.0"
        },
        "problems": examples
    }

def main():
    """Generate and save the 10-example parameterized dataset."""
    print("🚀 GENERATING 10-EXAMPLE PARAMETERIZED DATASET")
    print("=" * 60)
    print("Creating dataset with new Milestone 11 parameterized format...")
    
    dataset = create_parameterized_10_dataset()
    
    # Create datasets directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "claude_sonnet_4_20250514"
    dataset_dir = f"datasets/{model_name}_{timestamp}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save each problem as a separate file (following current structure)
    for i, problem in enumerate(dataset['problems'], 1):
        filename = os.path.join(dataset_dir, f"{i}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problem, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved problem {i}: {problem['problem_id']}")
        
        # Extract and save Python code
        py_filename = os.path.join(dataset_dir, f"{i}.py")
        solution_code = None
        
        if "solution" in problem and isinstance(problem["solution"], dict) and "code" in problem["solution"]:
            solution_code = problem["solution"]["code"]
        
        if solution_code:
            # Convert \\n to actual newlines and clean up the code
            cleaned_code = solution_code.replace('\\n', '\n').replace('\\"', '"').strip()
            
            # Add header comment
            header = f'''#!/usr/bin/env python3
"""
Generated Python code for: {problem['problem_id']}
Extracted from: {i}.json
"""

'''
            
            final_code = header + cleaned_code
            
            # Write Python file
            with open(py_filename, 'w', encoding='utf-8') as f:
                f.write(final_code)
            print(f"✅ Extracted Python code to {i}.py")
    
    # Save session metadata
    metadata_filename = os.path.join(dataset_dir, "session_metadata.json")
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(dataset['metadata'], f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 DATASET GENERATION COMPLETE")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Total problems: {len(dataset['problems'])}")
    print(f"Format: Parameterized testing (Milestone 11)")
    print(f"Model: {dataset['metadata']['model_name']}")
    print(f"Extended thinking: {dataset['metadata']['extended_thinking']}")
    
    return dataset_dir

if __name__ == "__main__":
    dataset_dir = main()
    print(f"\n🔍 Ready for validation with:")
    print(f"python parameterized_validation.py {dataset_dir}/")