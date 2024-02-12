from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

import numpy as np

from utils.timer import measure_execution_time

class Ansatz:
    def __init__(self, cfg):
        self.cfg = cfg['ansatz']
        self.molecule = cfg['molecule']
        self.model = cfg['generator']['model']
        self.encoding = cfg['generator']['encoding']
        
    @measure_execution_time
    def run(self, hamiltonian, n_qubits):
        # refer value
        if self.cfg['refereigensolver'] == 'NumPyMinimumEigensolver':
            refersolver = NumPyMinimumEigensolver()
            
        ref_value = refersolver.compute_minimum_eigenvalue(hamiltonian)
        print(f"Reference value: {ref_value.eigenvalue.real:.8f}")
        
        # VQE value
        n_particles = self.molecule['particle']
        num_particles = [n_particles // 2, n_particles // 2]
        
        if self.encoding == 'jordan_wigner' and self.model == 'qiskit':
            mapper = JordanWignerMapper()
            
        hf = HartreeFock(
            qubit_mapper=mapper, num_particles=num_particles, num_spatial_orbitals=n_qubits // 2)
        
        if self.cfg['ansatz'] == 'UCCSD':
            ansatz = UCCSD(
                qubit_mapper=mapper, num_particles=num_particles, num_spatial_orbitals=n_qubits // 2,
                initial_state=hf, generalized=False, preserve_spin=True)
            
        if self.cfg['optimizer'] == 'COBYLA':
            optimizer = COBYLA(maxiter=self.cfg['maxiter'])
            
        estimator = Estimator()
        
        vqe = VQE(estimator, ansatz=ansatz, optimizer=optimizer)
        vqe.initial_point = np.zeros(ansatz.num_parameters)
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        print(f"VQE (no noise): {result.optimal_value.real:.8f}")
        print(f"Delta from reference energy value is {(result.optimal_value.real - ref_value.eigenvalue.real):.8f}")
        
        return result
        