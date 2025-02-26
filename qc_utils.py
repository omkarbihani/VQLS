from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator,Pauli
from qiskit.circuit.gate import Gate
import numpy as np

from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.quantum_info import DensityMatrix

def get_LCU(obj):
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

def make_gate(obj, label=None):
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

def make_control_gate(obj, num_ctrl_qubits=1, label=None):
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


def hadamard_test(U_psi, Us, imag=False, draw_circuit=False, ideal=False, shots = 1024):
    """
    Perform the Hadamard test to estimate the real or imaginary part of a quantum expectation value.
    
    Parameters:
    U_psi (Gate): The initial unitary gate applied to the target qubits.
    Us (list of Gate): A list of unitary gates to be applied in the Hadamard test.
    imag (bool, optional): If True, measures the imaginary part. Default is False.
    draw_circuit (bool, optional): If True, prints the circuit diagram. Default is False.
    ideal (bool, optional): If True, computes the expectation value using the ideal density matrix. Default is False.
    
    Returns:
    float: The estimated expectation value.
    """
    if not isinstance(U_psi, Gate):
        raise TypeError("U_psi must be of type Gate")
    
    if not isinstance(Us, list) or not all(isinstance(U, Gate) and U.num_qubits == U_psi.num_qubits for U in Us):
        raise TypeError("Us must be a list of Gate objects with the same number of qubits as U_psi")
    
    n = U_psi.num_qubits + 1
    qc = QuantumCircuit(n, 1)
    
    qc.append(U_psi, list(range(1, n)))
    qc.barrier()
    
    qc.h(0)
    if imag:
        qc.sdg(0)
    qc.barrier()
    
    for U in Us:
        c_U_gate = make_control_gate(U, 1)
        qc.append(c_U_gate, list(range(n)))
    
    qc.barrier()
    qc.h(0)
    
    if ideal:
        if draw_circuit:
            print(qc.draw(fold=-1))

        dm = DensityMatrix(qc)
        probs_0 = dm.probabilities([0])
        expectation = probs_0[0] - probs_0[1]
   
    else:
        qc.measure(0, 0)
        
        if draw_circuit:
            print(qc.draw(fold=-1))
        
        backend = AerSimulator()
        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=0)
        qc_transpiled = pass_manager.run(qc)
        
        sampler = Sampler()
        result = sampler.run([qc_transpiled], shots=shots).result()
        counts = result[0].data.c.get_counts()
        expectation = (counts.get('0', 0) - counts.get('1', 0)) / shots

    return expectation


