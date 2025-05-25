from qiskit import QuantumCircuit, execute, Aer

def oracle(qc, params):
    """Oracle that marks the target state."""
    qc.z(qc.qubits[0])
    if params['marked_state'][0] == '1':
        qc.z(qc.qubits[0])
    if params['marked_state'][1] == '1':
        qc.z(qc.qubits[1])
    if params['marked_state'][2] == '1':
        qc.z(qc.qubits[2])
    if params['marked_state'][3] == '1':
        qc.z(qc.qubits[3])
    return qc

def diffusion(qc):
    """Diffusion operator to amplify the marked state."""
    qc.h([0, 1, 2, 3])
    qc.x([0, 1, 2, 3])
    qc.h(0)
    qc.cx([1, 2, 3], 0)
    qc.h(0)
    qc.x([0, 1, 2, 3])
    qc.h([0, 1, 2, 3])
    return qc

def grover(params):
    """Implement Grover's algorithm to find the marked state."""
    qc = QuantumCircuit(4)
    qc.h([0, 1, 2, 3])  # Prepare equal superposition
    
    for _ in range(params['num_iterations']):
        qc = oracle(qc, params)
        qc = diffusion(qc)
    
    qc.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Find the most frequent measurement, which is the marked state
    marked_state = max(counts, key=counts.get)
    return marked_state
