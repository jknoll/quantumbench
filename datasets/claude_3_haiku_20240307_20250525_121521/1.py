from qiskit import QuantumCircuit, execute, Aer
import numpy as np

def grover_algorithm(params):
    """Implement Grover's algorithm to find a marked state in a 4-qubit search space."""
    marked_state = params['marked_state']
    
    # Initialize the quantum circuit with 4 qubits
    qc = QuantumCircuit(4)
    
    # Apply the initial Hadamard gates to create the uniform superposition
    qc.h([0, 1, 2, 3])
    
    # Apply the Grover operator 3 times
    for _ in range(3):
        # Apply the oracle function that marks the target state
        qc.x(int(marked_state[0]))
        qc.x(int(marked_state[1]))
        qc.x(int(marked_state[2]))
        qc.x(int(marked_state[3]))
        qc.ccx(0, 1, 2)
        qc.x(int(marked_state[0]))
        qc.x(int(marked_state[1]))
        qc.x(int(marked_state[2]))
        qc.x(int(marked_state[3]))
        
        # Apply the diffusion operator
        qc.h([0, 1, 2, 3])
        qc.x([0, 1, 2, 3])
        qc.ccx(0, 1, 2)
        qc.x([0, 1, 2, 3])
        qc.h([0, 1, 2, 3])
    
    # Measure the final state
    qc.measure_all()
    
    # Execute the circuit on the simulator and get the results
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Determine the most likely measured state
    most_likely_state = max(counts, key=counts.get)
    
    # Check if the most likely state matches the marked state
    if most_likely_state == marked_state:
        success_probability = counts[most_likely_state] / 1024
        return success_probability
    else:
        return 0.0