#!/usr/bin/env python3
"""
Test Implementation Difference Detection

This demonstrates how the parameterized testing framework can detect
differences between correct and incorrect implementations of the same algorithm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameter_testing_framework import ParameterizedValidator, ValidationConfig

def main():
    """Test the framework's ability to detect implementation differences."""
    
    print("🔍 IMPLEMENTATION DIFFERENCE DETECTION TEST")
    print("=" * 60)
    print("Testing a correct vs. incorrect Grover implementation")
    print("to verify the framework can detect algorithmic differences.\n")
    
    # Correct Grover implementation
    correct_grover = """
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_qubits = params['n_qubits']
marked_states = params['marked_states']
iterations = params['optimal_iterations']

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

simulator = AerSimulator()
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()
"""

    # Incorrect implementation (missing diffusion operator)
    incorrect_grover = """
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

n_qubits = params['n_qubits']
marked_states = params['marked_states']
iterations = params['optimal_iterations']

qr = QuantumRegister(n_qubits, 'q')
cr = ClassicalRegister(n_qubits, 'c')
circuit = QuantumCircuit(qr, cr)

# Initialize superposition
circuit.h(qr)

# INCORRECT: Only apply oracle, missing diffusion operator
for _ in range(iterations):
    # Oracle for marked states
    for state in marked_states:
        binary = format(state, f'0{n_qubits}b')
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(qr[i])
        
        if n_qubits == 2:
            circuit.cz(qr[0], qr[1])
        elif n_qubits == 3:
            circuit.ccz(qr[0], qr[1], qr[2])
        
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(qr[i])
    
    # MISSING: Diffusion operator - this is the bug!

circuit.measure(qr, cr)

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
    
    print("📊 Testing Correct vs. Incorrect Implementation")
    print("-" * 50)
    
    # Test correct vs incorrect implementation
    results = validator.validate_algorithm(
        reference_code=correct_grover,
        target_code=incorrect_grover,
        algorithm_type='grover',
        n_trials=3
    )
    
    print(f"Algorithm Type: {results['algorithm_type']}")
    print(f"Total Parameter Tests: {results['total_tests']}")
    print(f"Successful Tests: {results['successful_tests']}")
    print(f"Success Rate: {results['success_rate']:.2f}")
    print(f"Meets Threshold (≥{config.min_success_rate}): {'✅ YES' if results['meets_threshold'] else '❌ NO'}")
    
    print(f"\nDetailed Results:")
    for i, test_result in enumerate(results['individual_results']):
        print(f"  Test {i+1}: {'✅ PASS' if test_result.success else '❌ FAIL'}")
        if not test_result.success and 'distribution' in test_result.details:
            tv_distance = test_result.details['distribution']['tv_distance']
            print(f"    - Total Variation Distance: {tv_distance:.3f}")
    
    print(f"\nTest Summary:")
    summary = results['summary']
    print(f"  • Success rate: {summary['success_rate']:.1%}")
    if summary['failure_reasons']:
        print(f"  • Failure reasons: {dict(summary['failure_reasons'])}")
    
    print(f"\n🎯 CONCLUSION")
    print("-" * 20)
    if not results['meets_threshold']:
        print("✅ SUCCESS: The framework correctly detected the algorithmic difference!")
        print("   The incorrect implementation (missing diffusion operator) was")
        print("   identified as statistically different from the correct one.")
        print("   This proves the system can distinguish good from bad implementations.")
    else:
        print("❌ UNEXPECTED: The framework did not detect the difference.")
        print("   This suggests the test parameters or tolerance may need adjustment.")
    
    print(f"\n💡 This demonstrates the framework's ability to catch implementation")
    print(f"   errors and validate correctness across different coding approaches!")

if __name__ == "__main__":
    main()