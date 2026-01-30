import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit.quantum_info import DensityMatrix

# ============================================================
# Parameters
# ============================================================

N = 4
t = 0.7
p = 1 - np.exp(-t)

shots = 2000

# ============================================================
# stochastic decay with probability
# ============================================================

def stochastic_reset_window(qc, targets, ancilla, p, ancilla_bit):
    theta = 2 * np.arcsin(np.sqrt(p))

    for q in targets:
        qc.ry(theta, ancilla)
        qc.measure(ancilla, ancilla_bit)
        qc.reset(q).c_if(ancilla_bit, 1)
        qc.reset(ancilla)

        
# ============================================================
# Circuit
# ============================================================

qc = QuantumCircuit(N+1, 6)
ancilla_bit = qc.clbits[0]

# initial state |0010>
qc.x(2)

# ---------------- Layer 1 ----------------

# decay on qubit [2]
stochastic_reset_window(qc, [2], ancilla=N, p=p, ancilla_bit=ancilla_bit)

qc.cx(2, 3)
qc.cx(2, 1)
qc.cx(1, 2)
qc.cx(1, 0)

# decay on qubits [0,1]
stochastic_reset_window(qc, [0, 1], ancilla=N, p=p, ancilla_bit=ancilla_bit)

qc.cx(0, 1)

qc.reset(0)

# ---------------- Layer 2 ----------------

qc.cx(3, 0)
qc.cx(3, 2)

# decay on qubits [2,3]
stochastic_reset_window(qc, [2, 3], ancilla=N, p=p, ancilla_bit=ancilla_bit)

qc.cx(2, 3)
qc.cx(2, 1)
qc.cx(1, 2)

qc.reset(1)

# ---------------- Layer 3 ----------------

# decay on qubit [0]
stochastic_reset_window(qc, [0], ancilla=N, p=p, ancilla_bit=ancilla_bit)

qc.cx(0, 1)
qc.cx(0, 3)
qc.cx(3, 0)
qc.cx(3, 2)

qc.measure([2,3], [1,2])   # data[0-1]

qc.reset(2)

qc.cx(1, 2)
qc.cx(1, 0)

qc.measure([0,1,2], [3,4,5])   # data[2-4]

# ============================================================
# Run on Aer density-matrix simulator
# ============================================================

sim = AerSimulator(method="density_matrix")

qc_t = transpile(qc, sim)

result = sim.run(qc_t, shots=shots).result()
counts = result.get_counts()

print("Counts:", counts)

# ============================================================
# Extract populations
# ============================================================

def data_from_counts(counts, bit_indices, total_bits):
    data = np.zeros(len(bit_indices))
    total = sum(counts.values())

    for bitstring, c in counts.items():
        # reverse to little-endian
        bits = bitstring[::-1]

        # pad with zeros for unmeasured bits
        bits = bits + "0" * (total_bits - len(bits))

        for j, b in enumerate(bit_indices):
            if bits[b] == "1":
                data[j] += c

    return data / total


# which classical bits store your 5 observables
bit_indices = [1, 2, 3, 4, 5]

data = data_from_counts(counts, bit_indices, total_bits=6)


print("Measured data:", data)
print("Reference exp(-t) ≈", np.exp(-t))
