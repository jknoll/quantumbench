from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
from collections import Counter

def create_grover_circuit():
    """Create Grover's algorithm circuit for finding |101⟩ in 3-qubit space"""
    # Create quantum and classical registers
    qreg = QuantumRegister(3, 'q')
    creg = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qreg, creg)
    
    # Step 1: Initialize superposition
    qc.h(qreg)  # Apply Hadamard to all qubits
    qc.barrier()
    
    # Step 2: Apply Grover iterations (optimal = 2 for N=8)
    num_iterations = 2
    
    for iteration in range(num_iterations):
        # Oracle for |101⟩: flip phase when q[0]=1, q[1]=0, q[2]=1
        qc.x(qreg[1])  # Flip qubit 1 to detect |0⟩ state
        qc.ccz(qreg[0], qreg[1], qreg[2])  # Multi-controlled Z
        qc.x(qreg[1])  # Restore qubit 1
        
        qc.barrier()
        
        # Diffusion operator (inversion about average)
        # H⊗3 · (2|000⟩⟨000| - I) · H⊗3
        qc.h(qreg)
        qc.x(qreg)  # Flip all qubits
        qc.ccz(qreg[0], qreg[1], qreg[2])  # Multi-controlled Z on |111⟩
        qc.x(qreg)  # Restore all qubits
        qc.h(qreg)
        
        qc.barrier()
    
    # Step 3: Measure all qubits
    qc.measure(qreg, creg)
    
    return qc

def analyze_results(counts, shots):
    """Analyze measurement results and calculate success probability"""
    # Convert to binary strings for analysis
    results = {}
    for state, count in counts.items():
        results[state] = count / shots
    
    # Check success probability for target state |101⟩
    target_state = '101'
    success_prob = results.get(target_state, 0)
    
    print(f"Measurement Results (probability):")
    for state in sorted(results.keys()):
        print(f"  |{state}⟩: {results[state]:.3f} ({counts.get(state, 0)} counts)")
    
    print(f"\nTarget state |{target_state}⟩ probability: {success_prob:.3f}")
    print(f"Theoretical optimal probability: ~0.894")
    print(f"Classical random search probability: 0.125")
    print(f"Quantum speedup factor: {success_prob/0.125:.2f}x")
    
    return success_prob

def calculate_theoretical_amplitudes():
    """Calculate theoretical amplitudes after 2 Grover iterations"""
    N = 8  # Total number of states
    marked_states = 1  # Number of marked states
    iterations = 2
    
    # Initial amplitude
    alpha_0 = 1/np.sqrt(N)
    
    # After k iterations
    theta = np.arcsin(np.sqrt(marked_states/N))
    final_amplitude = np.sin((2*iterations + 1) * theta)
    final_probability = final_amplitude**2
    
    print(f"Theoretical Analysis:")
    print(f"  Initial amplitude: {alpha_0:.3f}")
    print(f"  Final amplitude for |101⟩: {final_amplitude:.3f}")
    print(f"  Final probability for |101⟩: {final_probability:.3f}")
    
    return final_probability

# Main execution
if __name__ == "__main__":
    # Create the circuit
    grover_circuit = create_grover_circuit()
    
    print("Grover's Algorithm for 3-Qubit Search")
    print("Target state: |101⟩")
    print(f"Circuit depth: {grover_circuit.depth()}")
    print(f"Gate count: {grover_circuit.count_ops()}")
    
    # Theoretical analysis
    theoretical_prob = calculate_theoretical_amplitudes()
    
    # Simulate the circuit
    simulator = AerSimulator()
    shots = 8192
    
    # Transpile for simulator
    transpiled_circuit = transpile(grover_circuit, simulator)
    
    # Execute
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    print("\nSimulation Results:")
    success_prob = analyze_results(counts, shots)
    
    # Verify algorithm correctness
    print("\nVerification:")
    if abs(success_prob - theoretical_prob) < 0.05:
        print("✓ Algorithm working correctly - measured probability matches theory")
    else:
        print("⚠ Deviation from theoretical prediction detected")
    
    if success_prob > 0.125 * 3:  # At least 3x classical performance
        print("✓ Quantum speedup demonstrated")
    else:
        print("⚠ Quantum speedup not clearly demonstrated")