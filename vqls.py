from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
from qiskit.circuit.library import StatePreparation

from scipy.optimize import minimize
import numpy as np

import qc_utils as qcu

from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler

from abc import ABC, abstractmethod

import logging
import os
import sys

logger = logging.getLogger("vqls_logger")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

logger.addHandler(ch)


class LazyEvaluationAsString:
    """Class for lazy evaluation of function as string
    """
    def __init__(self, func: callable, *args, **kwargs):
        """Initialize the class.

        Args:
            func (callable): A function.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        """Return string representation of func output.
        """
        return(f'{self.func(*self.args, **self.kwargs)}')
 
class PrintInfo:
    """
    Base class for info/debug printing.
    """
    class _LazyStrings:
        """Helper class for lazy joining multiple strings. 
        """
        def __init__(self, *args):
            self.args = args

        def __str__(self):
            return(''.join((str(s) for s in self.args)))
        
    def who(self, back:int = 0 ) -> str:
        """Return caller information.

        Args:
            back (int, optional): Caller frame id. Defaults to 0.

        Returns:
            str: Caller information. 
        """ 
        frame = sys._getframe( back + 1 )
        
        return f'{os.path.basename( frame.f_code.co_filename )}, {frame.f_lineno}, {type(self).__name__}.{frame.f_code.co_name}()'

    def print_info(self, message:str = '', *args):
        """Print info message.

        Args:
            message (str): Prints info message.
        """
        logger.info(
            self._LazyStrings(
                f'{self.who(1)}: ', 
                message,
                *args
            )
        )
    
    def print_debug(self, message:str = '', *args):
        """Print debug message.

        Args:
            message (str): Prints debug message.
        """ 
        logger.debug(
            self._LazyStrings(
                f'{self.who(1)}: ', 
                message,
                *args
            )
        )

    def print_warning(self, message:str = '', *args):
        """Print warning message.

        Args:
            message (str): Prints warning message.
        """
        logger.warning(
            self._LazyStrings(
                f'{self.who(1)}: ', 
                message,
                *args
            )
        )
    
    def print_error(self, message:str = '', *args):
        """Print error message.

        Args:
            message (str): Prints debug error.
        """ 
        logger.error(
            self._LazyStrings(
                f'{self.who(1)}: ', 
                message,
                *args
            )
        )



class BaseVQLS(ABC, PrintInfo):
    """Abstract base class for Variational Quantum Linear Solvers."""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list|np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend=None,
    ):
        # --- basic attributes ---
        self.A = A
        self.b = b
        self.B = B
        self.d = d
        self.ansatz_function = ansatz_function
        self.params0 = params0
        self.backend = backend or AerSimulator()
       
        self.cost_history = []
        self._optimal_solution = None
        self._optimal_params = None

        # --- validation ---
        n = np.log2(self.A.shape[0])
        self.n_qubits = int(n)

        if 2**self.n_qubits != A.shape[0]:
            raise ValueError("Matrix A must be of size [2**n, 2**n].")
        if not np.allclose(A, A.T.conj(), atol=1e-6):
            raise ValueError("Matrix A must be Hermitian.")
        if b.shape[0] != 2**self.n_qubits:
            raise ValueError("Vector b must have length 2**n.")

        # --- LCU encoding ---
        self.A_LCU = A_LCU or qcu.get_LCU(A)

        if self.B is None:
            prep = StatePreparation(self.b)
            self.B = Operator(prep).to_matrix()

        self.B_LCU = B_LCU or qcu.get_LCU(self.B)


    @abstractmethod
    def calculate_cost(self,params):
        pass

    def get_optimal_solution(self):
        return self._optimal_solution
    
    def get_optimal_circuit(self):
        if self._optimal_solution is None:
            return None
        else:
            return self.ansatz_function(self.n_qubits, self.d, self._optimal_params)

    def run(self, initial_params: np.ndarray = None, method: str = "COBYLA", **kwargs):
        """
        Optimize variational parameters to minimize the cost.
        """

        if not initial_params:
            initial_params = self.params0

        self._optimal_solution = minimize(
            fun=lambda p: self.calculate_cost(p),
            x0=initial_params,
            method=method,
            **kwargs,
        )

        self._optimal_params = self._optimal_solution.x


class EstimatorVQLS(BaseVQLS):

    def _transpile(self, circuit: QuantumCircuit):
        """Transpile the given circuit using the preset pass manager."""
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=0)
        return pm.run(circuit)

    def _expectation(self, circuit: QuantumCircuit, operator: SparsePauliOp):
        """Compute expectation value <0| U^{dagger} O U |0>."""
        qc_transpiled = self._transpile(circuit)
        op_layout = operator.apply_layout(qc_transpiled.layout)
        estimator = Estimator()
        job = estimator.run([(qc_transpiled, op_layout)])
        result = job.result()[0]
        return result.data.evs


class EstimatorVQLSGlobal(EstimatorVQLS):

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list|np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend = None,
    ):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function, B=B,  A_LCU=A_LCU, B_LCU=B_LCU, backend=backend)
    

    def calculate_psi_norm(self, q_circuit: QuantumCircuit):
        """
        Compute <psi|psi> = <0|V^{dagger} A^{dagger} A V|0>.
        """
        op = self.A_LCU.adjoint() @ self.A_LCU
        return self._expectation(q_circuit, op)

    def calculate_inner_product_b_psi(self, q_circuit: QuantumCircuit):
        """
        Computes <b|psi> = <0|B^{dagger} A V|0>
        """
        V_LCU = qcu.get_LCU(q_circuit)
        M = Operator(self.B_LCU.adjoint() @ self.A_LCU @ V_LCU).data
        H_mat = 0.5 * (M + M.conj().T)
        K_mat = (M - M.conj().T) / (2j)

        H = SparsePauliOp.from_operator(Operator(H_mat))
        K = SparsePauliOp.from_operator(Operator(K_mat))

        qc = QuantumCircuit(self.A_LCU.num_qubits)

        real = self._expectation(qc, H)
        imag = self._expectation(qc, K)
        return real + 1j * imag

    def calculate_cost(self, params: np.ndarray):
        """
        Compute the global cost function:
        C_G = 1 - |<b|psi>|^2 / <psi|psi>
        """
        q_circuit = self.ansatz_function(self.n_qubits, self.d, params)
        denom = self.calculate_psi_norm(q_circuit)
        b_psi = self.calculate_inner_product_b_psi(q_circuit)
        num = np.abs(b_psi)**2

        cost = np.real(1 - (num / denom))
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost

        
class EstimatorVQLSLocal(EstimatorVQLS):

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list|np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend = None
    ):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function, B=B,  A_LCU=A_LCU, B_LCU=B_LCU, backend=backend)


    def calculate_psi_norm(self, q_circuit: QuantumCircuit):
        """
        Compute <psi|psi> = <0|V^{dagger} A^{dagger} A V|0>.
        """
        op = self.A_LCU.adjoint() @ self.A_LCU
        return self._expectation(q_circuit, op)

    def calculate_delta_j(self, j, q_circuit: QuantumCircuit):
        """ 
        delta_j = <0|V^{dagger} A^{dagger} B (Z_j otimes I_{bar{j}}) B^{dagger} A V|0>
        """

        sp = ''.join('Z' if i == j else 'I' for i in range(self.n_qubits))
        j_term = SparsePauliOp(sp)
        op = self.A_LCU.adjoint() @ self.B_LCU @ j_term @ self.B_LCU.adjoint() @ self.A_LCU
        # print(op)
        value = self._expectation(q_circuit, op)
        return value


    def calculate_cost(self, params: np.ndarray):
        """
        Compute the local cost function:
        C_L = 0.5 - (0.5 * sum_{j} <psi| A^{dagger} B (Z_j otimes I_{bar{j}}) B^{dagger} A |psi>) / (n * <psi|psi>)
        """
        q_circuit = self.ansatz_function(self.n_qubits, self.d, params)
        denom = self.calculate_psi_norm(q_circuit)
        
        num = 0
        for j in range(self.n_qubits):
            j_contribution = self.calculate_delta_j(j, q_circuit)
            # print(f"{j_contribution = }")
            num += j_contribution 
        
        cost = np.real(0.5 - (0.5*num /(self.n_qubits * denom)))
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost


# Sampler Classes defined below
class SamplerVQLS(BaseVQLS):

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list|np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend = None,
        num_shots: int = 1024
    ):
        
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function, B=B,  A_LCU=A_LCU, B_LCU=B_LCU, backend=backend)
        self.num_shots = num_shots 

    def _transpile(self, circuit: QuantumCircuit):
        """Transpile the given circuit using the preset pass manager."""
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=0)
        return pm.run(circuit)
    
    def _run_circuit(self, qc_transpiled):
        sampler = Sampler()
        result = sampler.run([qc_transpiled], shots=self.num_shots).result()[0]
        return result
    
    def _expectation(self, U0: Operator, operators: list[Operator | SparsePauliOp | Pauli]) -> complex:
        """
        Compute expectation value <0| U0^{dagger} Os U0 |0>.
        """
        self.print_debug()
        # Compute overall operator product
        composite_op = SparsePauliOp.from_operator(operators[0])
        for op in operators[1:]:
            composite_op = SparsePauliOp.from_operator(op) @ composite_op

        is_hermitian = np.allclose(composite_op.to_matrix(), composite_op.adjoint().to_matrix(), atol=1e-8)

        hadamard_test = qcu.SamplerHadamardTest(U_psi=U0, Us=operators, num_shots=self.num_shots)
        # hadamard_test = qcu.HadamardTest(U_psi=U0, Us=operators)
        exp_real = hadamard_test.get_expectation_real()
        exp_imag = 0 if is_hermitian else hadamard_test.get_expectation_imag()
        return exp_real + 1j * exp_imag
    

class SamplerVQLSGlobal(SamplerVQLS):

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list|np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend = None,
        num_shots: int = 1024
    ):
        
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function, B=B,  A_LCU=A_LCU, B_LCU=B_LCU, backend=backend, num_shots=num_shots)

        self.B_dagger_gate = qcu.make_gate(self.B_LCU.adjoint())

    def _get_beta_lm(self,qc_ansatz,l,m):
        """
        This function is used to calculate the terms for evaluating the norm of |psi> = A |tilde{x}> =  AV|0>, where V is ansatz. 
        A = sum_{m} a_l A_l.  
        <psi|psi> = sum_{lm} a_m* a_l beta_{lm}
        beta_{lm} = <0|V^T A_{m}^T A_{l} V |0>  
        """
            
        if (l == m):
            return 1
        
        V_operator = Operator(qc_ansatz)
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        return self._expectation(U0=V_operator, operators=[A_l, A_m.adjoint()])
        

    def _get_gamma_lm(self, qc_ansatz, l, m):
        """
        This function is used to calculate the terms in evaluating |<b|psi>|^2 = |<0|B^T A V |0>|^2 = sum_{lm} a_m* a_l gamma_{lm}. 
        A = sum_{m} a_l A_l.  
        gamma_{lm} = <0|B^T A_{l} V |0> <0|V^T A_{m}^T B |0> 
        """
        
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        V_operator = Operator(qc_ansatz)
        V_dagger_operator = V_operator.adjoint()

        

        if (l == m):
            
            # A_l_gate = qcu.make_gate(A_l)
            # qc = QuantumCircuit(self.n_qubits)
            # qc.append(V_operator,[i for i in range(self.n_qubits)])
            # qc.append(A_l_gate,[i for i in range(self.n_qubits)])
            # qc.append(self.B_dagger_gate,[i for i in range(self.n_qubits)])

            # qc.measure_all()

            # qc_transpiled = self._transpile(qc)
            # result = self._run_circuit(qc_transpiled)
            # gamma_lm = result.data.meas.get_counts().get('0'*self.n_qubits,0)/ self.num_shots
            

            I_operator = SparsePauliOp(''.join('I' for i in range(self.n_qubits)))
            gamma_l = self._expectation(U0=I_operator, operators=[V_operator, A_l, self.B_LCU.adjoint()])
            gamma_lm = float(gamma_l.real**2)

            return gamma_lm

        I_operator = SparsePauliOp(''.join('I' for i in range(self.n_qubits)))
        
        gamma_l = self._expectation(U0=I_operator, operators=[V_operator,A_l,self.B_LCU.adjoint()])
        gamma_m = self._expectation(U0=I_operator, operators=[self.B_LCU,A_m.adjoint(),V_dagger_operator])

        gamma_lm = gamma_l*gamma_m

        return gamma_lm

    def calculate_psi_norm(self, qc_ansatz):
        """ 
        This function evaluates the norm of the solution |psi>. |psi> = A|tilde{x}> = AV|0>. 
        <psi|psi> = sum_{lm} a_m* a_l beta_{lm}
        beta_{lm} = <0|V^T A_{m}^T A_{l} V |0> 
        """

        inner_product = 0
        for l in range(len(self.A_LCU)):
            for m in range(l,len(self.A_LCU)):
                beta_lm = self._get_beta_lm(qc_ansatz,l,m)
                inner_product += self.A_LCU.coeffs[m].conjugate()*self.A_LCU.coeffs[l]*beta_lm
                if l != m:
                    inner_product += self.A_LCU.coeffs[l].conjugate()*self.A_LCU.coeffs[m]*beta_lm
        
        return np.real(inner_product)

    def calculate_norm_b_psi_squared(self, qc_ansatz):
        """
        This function is used to evaluate |<b|psi>|^2 = |<0|B^T A V |0>|^2 = sum_{lm} a_m* a_l gamma_{lm}. 
        A = sum_{m} a_l A_l.  
        gamma_{lm} = <0|B^T A_{l} V |0> <0|V^T A_{m}^T B |0> 
        """

        inner_product = 0
        for l in range(len(self.A_LCU)):
            for m in range(len(self.A_LCU)):
                gamma_lm = self._get_gamma_lm(qc_ansatz,l,m)
                inner_product += self.A_LCU.coeffs[m].conjugate()*self.A_LCU.coeffs[l]*gamma_lm
                    
        return np.real(inner_product)
    
    def calculate_cost(self, params):
        """
        This function evaluates the global cost Hamiltonian.
        C_{G} = 1 - \frac{|<b|psi>|^2}{<psi|psi>}
        C_{G} = 1 - \frac{sum_{lm} a_l a_m* gamma_{lm}} {sum_{lm} a_l a_m* \beta_{lm}}
        """
        
        qc = self.ansatz_function(self.n_qubits, self.d, params)
        den = self.calculate_psi_norm(qc_ansatz=qc)
        # print(f"{den=}")
        num = self.calculate_norm_b_psi_squared(qc_ansatz=qc)
        # print(f"{num = }")

        cost = np.squeeze(1  - (num/den)).real
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost
        

class SamplerVQLSLocal(SamplerVQLS):

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list|np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend = None,
        num_shots: int = 1024
    ):

        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function, B=B,  A_LCU=A_LCU, B_LCU=B_LCU, backend=backend, num_shots=num_shots)

        self.B_dagger_gate = qcu.make_gate(self.B_LCU.adjoint())

    def _get_beta_lm(self,qc_ansatz,l,m):
        """
        This function is used to calculate the terms for evaluating the norm of |psi> = A |tilde{x}> =  AV|0>, where V is ansatz. 
        A = sum_{m} a_l A_l.  
        <psi|psi> = sum_{lm} a_m* a_l beta_{lm}
        beta_{lm} = <0|V^T A_{m}^T A_{l} V |0>  
        """
            
        if (l == m):
            return 1
        
        V_operator = Operator(qc_ansatz)
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        return self._expectation(U0=V_operator, operators=[A_l, A_m.adjoint()])
        

    def _get_delta_lm_j(self, qc_ansatz, l, m, j):
        """
        This function is used to calculate the terms in evaluating the local cost hamiltonian.
        delta_lm_j = <psi| A_m^{dagger} B (Z_j otimes I_{\bar_{j}}) B^{dagger} A |psi>
        """
        
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        V_operator = Operator(qc_ansatz)
        sp = ''.join('Z' if i == j else 'I' for i in range(self.n_qubits))
        j_term = SparsePauliOp(sp)
        operators = [A_l, self.B_LCU.adjoint(), j_term, self.B_LCU, A_m.adjoint()]
        delta_lm_j = self._expectation(U0=V_operator, operators=operators)

        return delta_lm_j

    def calculate_psi_norm(self, qc_ansatz):
        """ 
        This function evaluates the norm of the solution |psi>. |psi> = A|tilde{x}> = AV|0>. 
        <psi|psi> = sum_{lm} a_m* a_l beta_{lm}
        beta_{lm} = <0|V^T A_{m}^T A_{l} V |0> 
        """

        inner_product = 0
        for l in range(len(self.A_LCU)):
            for m in range(l,len(self.A_LCU)):
                beta_lm = self._get_beta_lm(qc_ansatz,l,m)
                inner_product += self.A_LCU.coeffs[m].conjugate()*self.A_LCU.coeffs[l]*beta_lm
                if l != m:
                    inner_product += self.A_LCU.coeffs[l].conjugate()*self.A_LCU.coeffs[m]*beta_lm
        
        return np.real(inner_product)


    def calculate_cost(self, params):
        """
        Compute the local cost function:
        C_L = 0.5 - (0.5 * sum_{j} <psi| A^{dagger} B (Z_j otimes I_{bar{j}}) B^{dagger} A |psi>) / (n * <psi|psi>)
        """
        
        self.print_debug()
        qc = self.ansatz_function(self.n_qubits, self.d, params)
        denom = self.calculate_psi_norm(qc_ansatz=qc)
        
        num = 0
        for j in range(self.n_qubits):
            for l in range(len(self.A_LCU)):
                for m in range(len(self.A_LCU)):
                    delta_lm_j = self._get_delta_lm_j(qc_ansatz=qc, l=l, m=m, j=j)
                    num += self.A_LCU.coeffs[l]*self.A_LCU.coeffs[m].conjugate()*delta_lm_j

        cost = np.real(0.5 - (0.5*num /(self.n_qubits * denom)))
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost
        
