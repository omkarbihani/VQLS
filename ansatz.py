from qiskit import QuantumCircuit
import numpy as np

def HE_ansatz(n: int, d: int, params: list) -> QuantumCircuit:
    """
    Creates a quantum circuit ansatz with rotational and entangling layers.
    
    Parameters:
    n (int): Number of qubits.
    d (int): Number of layers.
    params (list or np.ndarray): List of parameters with length 2 * n * d.
    
    Returns:
    QuantumCircuit: A Qiskit quantum circuit implementing the ansatz.
    
    Raises:
    AssertionError: If inputs are not of the expected type or length.
    """
    assert isinstance(n, int) and isinstance(d, int), "n and d must be integers"
    assert isinstance(params, (list, np.ndarray)), "params must be a list or numpy array"
    assert len(params) == 2 * n * d, "params must have length 2 * n * d"
    
    qc = QuantumCircuit(n)
    
    for i in range(d):
        # Apply rotation gates
        for j in range(n):
            qc.ry(params[2 * i * n + j], j)
            qc.rz(params[(2 * i + 1) * n + j], j)
        
        # Apply nearest-neighbor entangling CNOT gates
        for j in range(0, n - 1, 2):
            qc.cx(j, j + 1)
        for j in range(1, n - 1, 2):
            qc.cx(j, j + 1)
        
        # qc.barrier()
    
    return qc

def fixed_ansatz(n: int, d: int, params: list) -> QuantumCircuit:
    """
    Creates a quantum circuit ansatz with rotational and entangling layers.
    
    Parameters:
    n (int): Number of qubits.
    d (int): Number of layers.
    params (list or np.ndarray): List of parameters with length n * d.
    
    Returns:
    QuantumCircuit: A Qiskit quantum circuit implementing the ansatz.
    
    Raises:
    AssertionError: If inputs are not of the expected type or length.
    """
    assert isinstance(n, int) and isinstance(d, int), "n and d must be integers"
    assert isinstance(params, (list, np.ndarray)), "params must be a list or numpy array"
    assert len(params) == n * d, "params must have length n * d"
    
    qc = QuantumCircuit(n)
    
    for i in range(d):
        # Apply rotation gates
        for j in range(n):
            qc.ry(params[ i * n + j], j)
        
        # Apply nearest-neighbor entangling CNOT gates
        for j in range(0, n - 1, 2):
            qc.cz(j, j + 1)
        for j in range(1, n - 1, 2):
            qc.cz(j, j + 1)
        
        # qc.barrier()
    
    return qc