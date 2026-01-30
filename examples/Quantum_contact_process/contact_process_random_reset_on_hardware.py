import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime import QiskitRuntimeService   # ← uncomment for hardware

# ============================================================
# Parameters
# ============================================================

N = 4                  # system qubits
t = 0.7
p = 1 - np.exp(-t)     # decay probability per window
shots = 2000

# ============================================================
# Stochastic decay via ancilla + measurement + reset
# (hardware-compatible quantum trajectory)
# ============================================================

def stochastic_reset_window(qc, targets, ancilla, p, cbit):
    """
    Implements amplitude-damping *trajectory*:
    |1> -> |0> with probability p
    """
    theta = 2 * np.arcsin(np.sqrt(p))

    for q in targets:
        qc.ry(theta, ancilla)
        qc.measure(ancilla, cbit)
        qc.reset(q).c_if(cbit, 1)
        qc.reset(ancilla)

# ============================================================
# Circuit
# ============================================================

# +1 ancilla qubit, 6 classical bits for observables
qc = QuantumCircuit(N + 1, 6)

ancilla = N
ancilla_cbit = 0

# ------------------------------------------------------------
# Initial state |0010>
# ------------------------------------------------------------
qc.x(2)

# ======================== Layer 1 ===========================

qc.cx(2, 3)
qc.cx(2, 1)
qc.cx(1, 2)
qc.cx(1, 0)

stochastic_reset_window(qc, [0, 1], ancilla, p, ancilla_cbit)

qc.cx(0, 1)
qc.reset(0)

# ======================== Layer 2 ===========================

qc.cx(3, 0)
qc.cx(3, 2)

stochastic_reset_window(qc, [2, 3], ancilla, p, ancilla_cbit)

qc.cx(2, 3)
qc.cx(2, 1)
qc.cx(1, 2)

qc.reset(1)

# ======================== Layer 3 ===========================

qc.cx(0, 1)
qc.cx(0, 3)
qc.cx(3, 0)
qc.cx(3, 2)

# --- measure populations (trajectory-consistent) ---
qc.measure(2, 1)   # data[0]
qc.measure(3, 2)   # data[1]

qc.reset(2)

qc.cx(1, 2)
qc.cx(1, 0)

qc.measure(0, 3)   # data[2]
qc.measure(1, 4)   # data[3]
qc.measure(2, 5)   # data[4]

# ============================================================
# Run LOCALLY (hardware-style, shot-based)
# ============================================================

#sim = AerSimulator()
#qc_t = transpile(qc, sim)

#result = sim.run(qc_t, shots=shots).result()
#counts = result.get_counts()

#print("Counts:", counts)

# ============================================================
# Extract populations from counts
# ============================================================

def populations_from_counts(counts, bit_indices, total_bits):
    data = np.zeros(len(bit_indices))
    total = sum(counts.values())

    for bitstring, c in counts.items():
        bits = bitstring[::-1]    # little-endian fix
        bits = bits + "0" * (total_bits - len(bits))

        for j, b in enumerate(bit_indices):
            if bits[b] == "1":
                data[j] += c

    return data / total

bit_indices = [1, 2, 3, 4, 5]
data = populations_from_counts(counts, bit_indices, total_bits=6)

print("Measured data:", data)
print("Reference exp(-t) ≈", np.exp(-t))

# ============================================================
# To run on REAL hardware
# ============================================================

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")   # or ibm_kyoto, etc.

qc_hw = transpile(qc, backend, optimization_level=1)
job = backend.run(qc_hw, shots=shots)

result = job.result()
counts = result.get_counts()
print("Hardware counts:", counts)

