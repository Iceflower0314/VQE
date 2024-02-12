from qiskit.opflow import PauliSumOp

from openfermion.transforms import jordan_wigner
from utils.timer import measure_execution_time


@measure_execution_time
def Gene_Qiskit_hamiltonian(molecule_hamiltonian, cfg, n_qubits):
    if cfg['encoding'] == 'jordan_wigner':
        qubit_hamiltonian = jordan_wigner(molecule_hamiltonian)
        
    qubit_hamiltonian.compress()
    
    paulis = []
    coeffs = []

    for term, coeff in qubit_hamiltonian.terms.items():
        modes = [0] * n_qubits
        for qubit, op in term:
            if op == "X":
                modes[qubit] = 1
            elif op == "Y":
                modes[qubit] = 2
            elif op == "Z":
                modes[qubit] = 3
        pauli_str = "".join(["I", "X", "Y", "Z"][mode] for mode in reversed(modes))
        paulis.append(pauli_str)
        coeffs.append(coeff)

    pauli_op = [(pauli, weight) for pauli, weight in zip(paulis, coeffs)]
    hamiltonian = PauliSumOp.from_list(pauli_op)

    print(f"Number of qubits: {hamiltonian.num_qubits}")
    return hamiltonian