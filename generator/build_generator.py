from pyscf import M, ci

import openfermion.ops.representations as reps
from openfermion.config import EQ_TOLERANCE
import numpy as np

from generator.qiskit_gen import Gene_Qiskit_hamiltonian
from utils.timer import measure_execution_time

class Generator:
    def __init__(self, cfg):
        self.model = cfg['generator']['model']
        self.molecule = cfg['molecule']
        self.cfg = cfg['generator']
        self.n_qubits = 0
        
    @measure_execution_time
    def run(self):
        molecule_hamiltonian = self.gene_hamiltonian()
        if self.model == 'qiskit':
            return Gene_Qiskit_hamiltonian(molecule_hamiltonian, self.cfg, self.n_qubits)
            
    @measure_execution_time
    def gene_hamiltonian(self):
        m = M(atom=self.molecule['atoms'], basis="sto3g", 
              spin=self.molecule['spin'], charge=self.molecule['charge'])
        hf = m.RHF()
        hf.max_cycle = self.cfg['max_cycle'] # max number of iterations
        hf.conv_tol = self.cfg['conv_tol'] # converge threshold
        hf.kernel()
        
        for i in range(self.cfg['hf_cycle']):
            internal_stability_orbital = hf.stability()[0]
            if np.allclose(internal_stability_orbital, hf.mo_coeff):
                break
            dm = hf.make_rdm1(internal_stability_orbital, hf.mo_occ)
            hf = hf.run(dm)
            
        # ignore use_cas
        h1e = m.intor("int1e_kin") + m.intor("int1e_nuc")
        h2e = m.intor("int2e")
        scf_c = hf.mo_coeff
        nuclear_repulsion = m.energy_nuc()
        constant = nuclear_repulsion
        
        # Get the one and two electron integral in the Hatree Fock basis
        h1e = scf_c.T @ h1e @ scf_c
        for i in range(4):
            h2e = np.tensordot(h2e, scf_c, axes=1).transpose(3, 0, 1, 2)
        h2e = h2e.transpose(0, 2, 3, 1)
        
        myci = ci.CISD(hf).run()
        FCI_val = myci.e_tot
        print('_______cisd_______', FCI_val)
        
        core_adjustment, one_body_integrals, two_body_integrals = reps.get_active_space_integrals(
            h1e, h2e, self.molecule['occupied'], self.molecule['active'])
        constant += core_adjustment
        
        one_body_coefficients, two_body_coefficients = self.spinorb_from_spatial(
            one_body_integrals, two_body_integrals)
        molecular_hamiltonian = reps.InteractionOperator(
            constant, one_body_coefficients, 1 / 2 * two_body_coefficients)
        return molecular_hamiltonian
        
    def spinorb_from_spatial(self, one_body_integrals, two_body_integrals):
        n_qubits = 2 * one_body_integrals.shape[0]
        self.n_qubits = n_qubits
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
        
        for p in range(n_qubits//2):
            for q in range(n_qubits//2):
                one_body_coefficients[p, q] = one_body_integrals[p, q]
                one_body_coefficients[p + n_qubits // 2, q + n_qubits // 2] = one_body_integrals[p, q]
                
                for r in range(n_qubits//2):
                    for s in range(n_qubits//2):
                        # Mixed spin
                        two_body_coefficients[p, q + n_qubits // 2, r + n_qubits // 2, s] = (two_body_integrals[p, q, r, s])
                        two_body_coefficients[p + n_qubits // 2,  q, r, s +
                                          n_qubits // 2] = (two_body_integrals[p, q, r, s])

                        # Same spin
                        two_body_coefficients[p, q, r, s] = (two_body_integrals[p, q, r, s])
                        two_body_coefficients[p + n_qubits // 2, q + n_qubits // 2, r + n_qubits // 2, s +
                                          n_qubits // 2] = (two_body_integrals[p, q, r, s])
        
        # Truncate.
        one_body_coefficients[
            np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
        two_body_coefficients[
            np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.
        return one_body_coefficients, two_body_coefficients