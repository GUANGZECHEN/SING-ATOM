import numpy as np
from functools import reduce
from qiskit.quantum_info import Operator, Pauli, Statevector, DensityMatrix

# ------------------------
# Basic single-qubit ops
# ------------------------

sx = Operator(Pauli("X"))
sy = Operator(Pauli("Y"))
sz = Operator(Pauli("Z"))
id2 = Operator(np.eye(2))

sp = (sx + 1j * sy) / 2
sm = (sx - 1j * sy) / 2

n_op = ( -sz + id2) / 2          # population operator


# ------------------------
# Tensor helpers
# ------------------------

def tensor_ops(op_list):
    return reduce(lambda a, b: a.tensor(b), op_list)

def one_qubit_op(A, m, N):
    """Operator A acting on qubit m (0-indexed)"""
    ops = []
    for j in range(N):
        ops.append(A if j == m else id2)
    return tensor_ops(ops)

def CX_mn(m, n, N):
    """Projector-based CNOT: control m -> target n"""
    P0 = (sz + id2) / 2
    P1 = (-sz + id2) / 2

    term1 = one_qubit_op(P1, m, N) @ one_qubit_op(sx, n, N)
    term2 = one_qubit_op(P0, m, N)
    return term1 + term2


# ------------------------
# Lindblad evolution
# ------------------------

def amplitude_damp(rho, q, N, t):
    p = 1 - np.exp(-t)

    K0 = np.array([[1, 0],
                   [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)],
                   [0, 0]], dtype=complex)

    K0_full = one_qubit_op(Operator(K0), q, N).data
    K1_full = one_qubit_op(Operator(K1), q, N).data

    return (
        K0_full @ rho @ K0_full.conj().T
        + K1_full @ rho @ K1_full.conj().T
    )

def decay_layer(rho, qubits, N, t):
    for q in qubits:
        rho = amplitude_damp(rho, q, N, t)
    return rho

# ------------------------
# Reset channel
# ------------------------

def reset_qubit(rho, q, N):
    K0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)
    K1 = np.array([[0, 1],
                   [0, 0]], dtype=complex)

    K0_full = one_qubit_op(Operator(K0), q, N).data
    K1_full = one_qubit_op(Operator(K1), q, N).data

    return (
        K0_full @ rho @ K0_full.conj().T
        + K1_full @ rho @ K1_full.conj().T
    )

# ------------------------
# Expectation values
# ------------------------

def populations(rho, N):
    pops = []
    for i in range(N):
        ni = one_qubit_op(n_op, i, N).data
        pops.append(np.real(np.trace(ni @ rho)))
    return pops

# ========================
# Simulation parameters
# ========================

N = 4
t = 0.7

# initial state |0010>
from qiskit.quantum_info import Statevector

# |q3 q2 q1 q0⟩ = |0 0 1 0⟩
psi0 = Statevector.from_label("0010")
rho = DensityMatrix(psi0).data


print("Initial populations:", populations(rho, N))

# ========================
# Layer 1
# ========================

rho = decay_layer(rho, [2], N, t)

rho = CX_mn(2, 3, N).data @ rho @ CX_mn(2, 3, N).data.conj().T
rho = CX_mn(2, 1, N).data @ rho @ CX_mn(2, 1, N).data.conj().T
rho = CX_mn(1, 2, N).data @ rho @ CX_mn(1, 2, N).data.conj().T
rho = CX_mn(1, 0, N).data @ rho @ CX_mn(1, 0, N).data.conj().T

print("After branching L1:", populations(rho, N))

rho = decay_layer(rho, [0,1], N, t)
print("After dissipation L1:", populations(rho, N))

rho = CX_mn(0, 1, N).data @ rho @ CX_mn(0, 1, N).data.conj().T
print("After CNOT L1:", populations(rho, N))

rho = reset_qubit(rho, 0, N)
print("Layer 1 finished:", populations(rho, N))

# ========================
# Layer 2
# ========================

rho = CX_mn(3, 0, N).data @ rho @ CX_mn(3, 0, N).data.conj().T
rho = CX_mn(3, 2, N).data @ rho @ CX_mn(3, 2, N).data.conj().T

rho = decay_layer(rho, [2,3], N, t)
print("After dissipation L2:", populations(rho, N))

rho = CX_mn(2, 3, N).data @ rho @ CX_mn(2, 3, N).data.conj().T
rho = CX_mn(2, 1, N).data @ rho @ CX_mn(2, 1, N).data.conj().T
rho = CX_mn(1, 2, N).data @ rho @ CX_mn(1, 2, N).data.conj().T

rho = reset_qubit(rho, 1, N)
print("Layer 2 finished:", populations(rho, N))

# ========================
# Layer 3
# ========================

rho = decay_layer(rho, [0], N, t)

rho = CX_mn(0, 1, N).data @ rho @ CX_mn(0, 1, N).data.conj().T
rho = CX_mn(0, 3, N).data @ rho @ CX_mn(0, 3, N).data.conj().T
rho = CX_mn(3, 0, N).data @ rho @ CX_mn(3, 0, N).data.conj().T
rho = CX_mn(3, 2, N).data @ rho @ CX_mn(3, 2, N).data.conj().T

data = np.zeros(5)
data[0] = populations(rho, N)[2]
data[1] = populations(rho, N)[3]

rho = reset_qubit(rho, 2, N)
print("Layer 3 finished:", populations(rho, N))

rho = CX_mn(1, 2, N).data @ rho @ CX_mn(1, 2, N).data.conj().T
rho = CX_mn(1, 0, N).data @ rho @ CX_mn(1, 0, N).data.conj().T

pops = populations(rho, N)
data[2] = pops[0]
data[3] = pops[1]
data[4] = pops[2]

print("\nFinal data:", data)
print("Expected transition ~ exp(-0.7) ≈", np.exp(-0.7))

