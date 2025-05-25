#!/usr/bin/env python3
"""
Demo: Parameterized Testing for Implementation-Independent Validation

This demonstrates how the new parameterized testing framework can validate
different implementations of the same quantum algorithm against each other,
enabling robust evaluation of LLM-generated quantum code.
"""

from parameter_testing_framework import ParameterizedValidator, ValidationConfig

def main():
    """Demonstrate parameterized testing with different Grover implementations."""
    
    print("🔬 PARAMETERIZED TESTING DEMONSTRATION")
    print("=" * 60)
    print("Testing different Grover's algorithm implementations using")
    print("the same parameter sets to validate implementation-independence.\n")
    
    # Reference implementation (simplified)
    reference_grover = """
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# Extract parameters
n_qubits = params['n_qubits']
marked_states = params['marked_states']
iterations = params['optimal_iterations']

# Create circuit
qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Initialize superposition
circuit.h(qr)

# Grover iterations
for _ in range(iterations):
    # Oracle for marked states
    for state in marked_states:
        binary = format(state, f'0{n_qubits}b')
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(qr[i])
        
        # Multi-controlled Z
        if n_qubits == 2:
            circuit.cz(qr[0], qr[1])
        elif n_qubits == 3:
            circuit.ccz(qr[0], qr[1], qr[2])
        
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(qr[i])
    
    # Diffusion operator
    circuit.h(qr)
    circuit.x(qr)
    
    if n_qubits == 2:
        circuit.cz(qr[0], qr[1])
    elif n_qubits == 3:
        circuit.ccz(qr[0], qr[1], qr[2])
    
    circuit.x(qr)
    circuit.h(qr)

circuit.measure(qr, cr)

# Execute
simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()
"""

    # Alternative implementation (different structure but same algorithm)
    alternative_grover = """
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

def create_oracle(circuit, qubits, marked_states):
    for state in marked_states:
        binary = format(state, f'0{len(qubits)}b')
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(qubits[i])
        
        if len(qubits) == 2:
            circuit.cz(qubits[0], qubits[1])
        elif len(qubits) == 3:
            circuit.ccz(qubits[0], qubits[1], qubits[2])
        
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(qubits[i])

def create_diffusion(circuit, qubits):
    circuit.h(qubits)
    circuit.x(qubits)
    
    if len(qubits) == 2:
        circuit.cz(qubits[0], qubits[1])
    elif len(qubits) == 3:
        circuit.ccz(qubits[0], qubits[1], qubits[2])
    
    circuit.x(qubits)
    circuit.h(qubits)

# Extract parameters
n_qubits = params['n_qubits']
marked_states = params['marked_states']
iterations = params['optimal_iterations']

# Create circuit using modular approach
qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Initialize
circuit.h(qr)

# Apply Grover operator
for _ in range(iterations):
    create_oracle(circuit, qr, marked_states)
    create_diffusion(circuit, qr)

circuit.measure(qr, cr)

# Execute
simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()
"""

    # Test configuration
    config = ValidationConfig(
        shots=1000,
        tolerance=0.05,
        min_success_rate=0.8
    )
    
    validator = ParameterizedValidator(config)
    
    print("📊 Testing Implementation Equivalence")
    print("-" * 40)
    
    # Test with Grover's algorithm
    results = validator.validate_algorithm(
        reference_code=reference_grover,
        target_code=alternative_grover,
        algorithm_type='grover',
        n_trials=3
    )
    
    print(f"Algorithm Type: {results['algorithm_type']}")
    print(f"Total Parameter Tests: {results['total_tests']}")
    print(f"Successful Tests: {results['successful_tests']}")
    print(f"Success Rate: {results['success_rate']:.2f}")
    print(f"Meets Threshold (≥{config.min_success_rate}): {'✅ YES' if results['meets_threshold'] else '❌ NO'}")
    
    print(f"\nTest Summary:")
    summary = results['summary']
    print(f"  • Total tests: {summary['total_tests']}")
    print(f"  • Successful: {summary['successful_tests']}")
    print(f"  • Failed: {summary['failed_tests']}")
    print(f"  • Success rate: {summary['success_rate']:.1%}")
    
    if summary['failure_reasons']:
        print(f"  • Failure reasons: {dict(summary['failure_reasons'])}")
    
    print(f"\n🎯 CONCLUSION")
    print("-" * 20)
    if results['meets_threshold']:
        print("✅ The alternative implementation is statistically equivalent")
        print("   to the reference implementation across different parameters.")
        print("   This demonstrates successful implementation-independent validation!")
    else:
        print("❌ The implementations show significant statistical differences.")
        print("   This indicates either algorithmic errors or parameter sensitivity.")
    
    print(f"\n💡 This approach enables validation of any LLM-generated code")
    print(f"   against reference implementations without requiring exact")
    print(f"   output matching - perfect for fine-tuning evaluation!")

if __name__ == "__main__":
    main()