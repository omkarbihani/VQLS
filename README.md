# Implementation of VQLS

## Theory Overview
The goal of the **Variational Quantum Linear Solver (VQLS)** is to solve the linear system of equations:

$$|x\rangle = \frac{A^{-1} |b\rangle}{ ||A^{-1} |b\rangle|| }$$

We approximate $|x\rangle \approx |\tilde{x}\rangle$.
$$|\tilde{x}\rangle = V(\theta) |0\rangle$$

Where $V(\theta)$ represents a **variational ansatz** (a parameterized quantum circuit). The vector $|\tilde{x}\rangle$ is the approximate solution of the linear system.

### Key Notations:
- $|\psi\rangle = A|\tilde{x}\rangle$: This is the intermediate state produced by applying matrix $A$ to the ansatz.
- $A = \sum_{l} c_l A_l$: Matrix $A$ is expressed as a linear combination of operators $A_l$ with coefficients $c_l$.
- $|b\rangle = B|0\rangle$: The target state $|b\rangle$ is related to the identity operator $B$ applied to the quantum state $|0\rangle$.


### Global Cost Hamiltonian

To find the optimal parameters for the variational ansatz, we minimize the **global cost Hamiltonian** defined as:

$$H_{G} = A^{\dagger} (\mathbb{1} - |b\rangle \langle b|) A$$

The corresponding global cost function is:

$$C_{G} = \frac{\langle \psi| H_{G} | \psi \rangle}{\langle \psi| \psi\rangle} = 1 - \frac{| \langle b | \psi \rangle |^2}{\langle \psi| \psi\rangle}$$

Where:
- $\langle \psi| \psi\rangle = \langle \tilde{x}| A^{\dagger} A |\tilde{x}\rangle = \sum_{l,m} c_{m}^{*} c_{l} \beta_{lm}$, with $\beta_{lm} = \langle \tilde{x}| A_{m}^{\dagger} A_{l} |\tilde{x}\rangle$
- $| \langle b | \psi \rangle |^2 = \langle \psi | b \rangle \langle b | \psi \rangle = \sum_{l,m} c_m^{*} c_{l} \gamma_{lm}$, with $\gamma_{lm} = \langle 0| V^{\dagger} A_{m}^{\dagger} B | 0 \rangle \langle 0 | B^{\dagger} A_l V|0 \rangle$


### Local Cost Hamiltonian

An alternative local cost function based on the local Hamiltonian is:

$$H_{L} = A^{\dagger} B \left( \mathbb{1} - \frac{1}{n} \sum_{j=1}^{n} |0_j\rangle \langle 0_j | \otimes \mathbb{1}_{\bar{j}} \right) B^{\dagger} A$$

This gives the local cost:

$$C_{L} = \frac{\langle \psi| H_{L} | \psi \rangle}{\langle \psi| \psi\rangle} $$

We can write $|0_j\rangle \langle 0_j | = \frac{Z_j + \mathbb{1}_{j}}{2}$. After simplification

$$C_{L} = \frac{1}{2} - \frac{1}{2n} \sum_{j} \langle \psi | A^{\dagger} B \left( Z_{j} \otimes \mathbb{1}_{\bar{j}} \right) B^{\dagger} A | \psi \rangle$$

$$\langle \psi | A^{\dagger} B \left( Z_{j} \otimes \mathbb{1}_{\bar{j}} \right) B^{\dagger} A | \psi \rangle = \sum_{l,m} c_m^{*} c_l \delta_{lm}^{(j)} $$

Where:
- $\delta_{lm}^{(j)} = \langle \psi | A_m^{\dagger} B \left( Z_{j} \otimes \mathbb{1}_{\bar{j}} \right) B^{\dagger} A_l | \psi \rangle$


## Implementation Details

**Code Structure and Files:**

1. **ansatz.py**:
    - We use parameterized quantum circuits to represent the variational ansatz. The ansatz is built using either the `HE_ansatz` or `fixed_ansatz` functions that define rotation and entangling layers for the quantum circuit.

2. **qc_utils.py**:
    - Contains fucnctions for creating gates and their controlled version.  **Linear Combination of Unitaries (LCU)** representation using `get_LCU` function.
    - Has classes defined for Hadamard test.

3. **vqls.py**:
    - Estimator-based VQLS: Uses `EstimatorV2` to evaluate expectation values based on quantum circuits for local and gloabl cost functions.
    - Sampler-based VQLS: Uses `SamplerV2` for quantum state sampling to estimate the expectation values for local and gloabl cost functions.

4. **test.ipynb**:
    - Contains examples.

### Reference

**Bravo-Prieto, C., LaRose, R., Cerezo, M., Subasi, Y., Cincio, L., & Coles, P. J. (2023).** *Variational Quantum Linear Solver*. *Quantum*, 7, 1188. [https://doi.org/10.22331/q-2023-11-22-1188](https://doi.org/10.22331/q-2023-11-22-1188)

