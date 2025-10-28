from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator,Pauli
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Statevector
import numpy as np

from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as Sampler

def get_LCU(obj) -> SparsePauliOp:
    """
    Convert an input object into a Linear Combination of Unitaries (LCU) representation.
    
    Parameters:
    obj (np.ndarray, QuantumCircuit, SparsePauliOp, Operator): The input object to convert.
    
    Returns:
    SparsePauliOp: The LCU representation of the input object.
    """
    if isinstance(obj, np.ndarray):
        return SparsePauliOp.from_operator(Operator(obj))
    
    if isinstance(obj, QuantumCircuit):
        return SparsePauliOp.from_operator(Operator.from_circuit(obj))
    
    if isinstance(obj, (SparsePauliOp, Operator)):
        return SparsePauliOp.from_operator(obj)
    
    raise TypeError("Input must be of type np.ndarray, QuantumCircuit, SparsePauliOp, or Operator.")

def make_gate(obj, label=None) -> Gate:
    """
    Convert an input object into a Qiskit gate.
    
    Parameters:
    obj (QuantumCircuit, np.ndarray, Operator, SparsePauliOp, Pauli, Gate): The input object.
    label (str, optional): Label for the created gate.
    
    Returns:
    Gate: A Qiskit gate representation of the input object.
    """
    if isinstance(obj, QuantumCircuit):
        return obj.to_gate(label=label)
    
    if isinstance(obj, (np.ndarray, Operator, SparsePauliOp, Pauli)):
        op = Operator(obj)
        if not op.is_unitary():
            raise ValueError("Input object must be a unitary matrix.")
        
        qc = QuantumCircuit(op.num_qubits)
        qc.unitary(op, list(range(op.num_qubits)))
        return qc.to_gate(label=label)
    
    if isinstance(obj, Gate):
        if label:
            obj.name = label
        return obj
    
    raise TypeError("Input must be of type QuantumCircuit, np.ndarray, Operator, SparsePauliOp, Pauli, or Gate.")

def make_control_gate(obj, num_ctrl_qubits=1, label=None) -> Gate:
    """
    Create a controlled version of a given quantum gate.
    
    Parameters:
    obj (QuantumCircuit, np.ndarray, Operator, SparsePauliOp, Pauli, Gate): The input object.
    num_ctrl_qubits (int, optional): Number of control qubits. Default is 1.
    label (str, optional): Label for the controlled gate.
    
    Returns:
    Gate: A controlled version of the input gate.
    """
    gate = make_gate(obj)
    return gate.control(num_ctrl_qubits, label=label)



class HadamardTest():
    def __init__(self,
                 U_psi: SparsePauliOp | Operator | Pauli,
                 Us: list[SparsePauliOp | Operator | Pauli]):
        
        """
        Perform the Hadamard test to estimate the real or imaginary part of a quantum expectation value.
        
        Parameters:
        U_psi (Gate): The initial unitary gate applied to the target qubits.
        Us (list of Gate): A list of unitary gates to be applied in the Hadamard test.
        """
        
        self.U_psi = U_psi
        self.Us = Us

        if not isinstance(U_psi, SparsePauliOp | Operator | Pauli):
            raise TypeError("U_psi must be of type SparsePauliOp")
    
        if not isinstance(Us, list):
            raise TypeError("Us must be a list")
        
        if not all(isinstance(U, SparsePauliOp | Operator | Pauli) for U in Us):
            raise TypeError("Us must contain SparsePauliOp or Operator objects")
        
        if  not all( U.num_qubits == U_psi.num_qubits for U in Us):
            raise TypeError("Us must be a list of SparsePauliOp objects with the same number of qubits as U_psi")
        self._n_qubits = self.U_psi.num_qubits + 1

    
    def _circuit(self, imag=False):
        qc = QuantumCircuit(self._n_qubits)
        U_psi_gate = make_gate(self.U_psi)
        qc.append(U_psi_gate, list(range(1, self._n_qubits)))

        qc.h(0)
        if imag:
            qc.sdg(0)
        qc.barrier()

        for U in self.Us:
            c_U_gate = make_control_gate(U, 1)
            qc.append(c_U_gate, list(range(self._n_qubits)))
        
        qc.barrier()
        qc.h(0)

        return qc
    
    def get_expectation_real(self):
        qc = self._circuit(imag=False)
        sv = Statevector(qc)
        probs = sv.reverse_qargs().probabilities_dict()
        p0 = sum([probs[i] for i in probs.keys() if i[0]=='0'])
        p1 = sum([probs[i] for i in probs.keys() if i[0]=='1'])
        value = p0 - p1

        return value
    
    def get_expectation_imag(self):
        qc = self._circuit(imag=True)
        sv = Statevector(qc)
        probs = sv.reverse_qargs().probabilities_dict()
        p0 = sum([probs[i] for i in probs.keys() if i[0]=='0'])
        p1 = sum([probs[i] for i in probs.keys() if i[0]=='1'])
        value = p0 - p1

        return value


class SamplerHadamardTest(HadamardTest):

    def __init__(self, U_psi, Us, backend=AerSimulator(), num_shots=1024):
        super().__init__(U_psi, Us)
        self.backend = backend
        self.num_shots = num_shots

    def _circuit(self, imag=False):
        qc = QuantumCircuit(self._n_qubits, 1)
        U_psi_gate = make_gate(self.U_psi)
        qc.append(U_psi_gate, list(range(1, self._n_qubits)))

        qc.h(0)
        if imag:
            qc.sdg(0)
        qc.barrier()

        for U in self.Us:
            c_U_gate = make_control_gate(U, 1)
            qc.append(c_U_gate, list(range(self._n_qubits)))
        
        qc.barrier()
        qc.h(0)
        qc.measure(0, 0)

        return qc
    
    def _transpile(self, qc):
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=0)
        return pm.run(qc)
    
    def _run_circuit(self, qc_transpiled):
        sampler = Sampler()
        result = sampler.run([qc_transpiled], shots=self.num_shots).result()
        return result

    def get_expectation_real(self):
        qc = self._circuit(imag=False)

        qc_transpiled = self._transpile(qc)
        result = self._run_circuit(qc_transpiled)
        counts = result[0].data.c.get_counts()
        value = (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots

        return value
    
    def get_expectation_imag(self):
        qc = self._circuit(imag=True)

        qc_transpiled = self._transpile(qc)
        result = self._run_circuit(qc_transpiled)
        counts = result[0].data.c.get_counts()
        value = (counts.get('0', 0) - counts.get('1', 0)) / self.num_shots

        return value
