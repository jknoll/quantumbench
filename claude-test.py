import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def create_grover_circuit(target_state='101'):
    """
    Create Grover's algorithm circuit to search for target_state.
    
    Args:
        target_state (str): Binary string representing target state
    
    Returns:
        QuantumCircuit: Complete Grover circuit
    """
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Step 1: Initialize uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()
    
    # Calculate optimal iterations
    N = 2**n_qubits  # Total states
    optimal_iterations = int(np.pi/4 * np.sqrt(N))
    print(f"Optimal iterations for {N} states: {optimal_iterations}")
    
    # Step 2: Apply Grover operator (Oracle + Diffusion) optimal times
    for iteration in range(optimal_iterations):
        # Oracle: flip phase of target state |101⟩
        oracle_circuit = create_oracle(target_state)
        qc.compose(oracle_circuit, inplace=True)
        qc.barrier()
        
        # Diffusion operator (amplitude amplification)
        diffusion_circuit = create_diffusion_operator(n_qubits)
        qc.compose(diffusion_circuit, inplace=True)
        qc.barrier()
    
    # Step 3: Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc

def create_oracle(target_state):
    """
    Create oracle that flips phase of target state.
    For |101⟩: flip phase when q0=1, q1=0, q2=1
    """
    n_qubits = len(target_state)
    oracle = QuantumCircuit(n_qubits)
    
    # Flip qubits that should be 0 in target state
    for i, bit in enumerate(target_state):
        if bit == '0':
            oracle.x(i)
    
    # Multi-controlled Z gate (phase flip when all qubits are |1⟩)
    if n_qubits == 3:
        oracle.ccz(0, 1, 2)  # 3-qubit controlled-Z
    else:
        # General case: multi-controlled Z
        oracle.h(n_qubits-1)
        oracle.mct(list(range(n_qubits-1)), n_qubits-1)
        oracle.h(n_qubits-1)
    
    # Flip back the qubits we flipped
    for i, bit in enumerate(target_state):
        if bit == '0':
            oracle.x(i)
    
    return oracle

def create_diffusion_operator(n_qubits):
    """
    Create diffusion operator: 2|s⟩⟨s| - I
    where |s⟩ is uniform superposition state
    """
    diffusion = QuantumCircuit(n_qubits)
    
    # Transform |s⟩ to |0...0⟩
    diffusion.h(range(n_qubits))
    
    # Flip phase of |0...0⟩ state
    diffusion.x(range(n_qubits))
    
    # Multi-controlled Z gate
    if n_qubits == 3:
        diffusion.ccz(0, 1, 2)
    else:
        diffusion.h(n_qubits-1)
        diffusion.mct(list(range(n_qubits-1)), n_qubits-1)
        diffusion.h(n_qubits-1)
    
    diffusion.x(range(n_qubits))
    
    # Transform back to |s⟩
    diffusion.h(range(n_qubits))
    
    return diffusion

# Execute the algorithm
target = '101'
grover_circuit = create_grover_circuit(target)

print(f"Searching for state: |{target}⟩")
print(f"Circuit depth: {grover_circuit.depth()}")
print(f"Gate count: {grover_circuit.count_ops()}")

# Simulate
simulator = AerSimulator()
transpiled_circuit = transpile(grover_circuit, simulator)
job = simulator.run(transpiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()

# Display results
print("\nMeasurement results:")
for state, count in sorted(counts.items()):
    probability = count/1000
    print(f"|{state}⟩: {count} counts ({probability:.3f} probability)")

# Theoretical success probability after 2 iterations
theta = np.arcsin(1/np.sqrt(8))  # angle for 1 target in 8 states
success_prob_theory = np.sin((2*2 + 1) * theta)**2
print(f"\nTheoretical success probability: {success_prob_theory:.3f}")
print(f"Observed success probability: {counts.get(target, 0)/1000:.3f}")