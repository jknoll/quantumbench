from qiskit import QuantumCircuit, Aer
from qiskit_aqua.algorithms import QAOA
from qiskit_aqua.components.optimizers import SPSA
from qiskit_aqua.operators import WeightedPauliOperator

def main():
    # Define the graph edges
    edges = [(0, 1), (1, 2), (0, 2)]
    n_qubits = 3

    # Create the Ising Hamiltonian for Max-Cut
    pauli_list = []
    for u, v in edges:
        if u < v:  # Avoid duplicate edges
            term = ((u, 'Z'), (v, 'Z'))
            pauli_list.append((term, 1.0))

    wpo = WeightedPauliOperator.from_terms(pauli_list)

    # Set up QAOA with 1 repetition and suitable mixer
    qaoa = QAOA(wpo, p_classical=1, quantum_instance=Aer.get_backend('qasm_simulator'),
                 optimizer=SPSA(maxiter=50), mixer=TwoLocal(n_qubits, ['ry', 'rz'], 'esz'))

    # Run QAOA
    result = qaoa.run()

    # Get the optimal parameters (gamma and beta)
    optimal_params = result.x

    # Build the optimal circuit
    optimal_circuit = result.circuit.bind_parameters(optimal_params)
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(optimal_circuit, shots=1000)
    counts = job.result().get_counts()

    # Determine the most probable bitstring
    bitstr = max(counts, key=lambda k: counts[k])
    print(f"Optimal cut partition: {bitstr}")

    # Compute the Max-Cut value
    def max_cut_value(bitstr):
        cut = 0
        for u, v in edges:
            cut += (bitstr[u] != bitstr[v])
        return cut

    max_cut = max_cut_value(bitstr)
    print(f"Max cut value: {max_cut}")

    return result

if __name__ == "__main__":
    result = main()
    # Example output: partition [0, 1] and [2] with value 2
