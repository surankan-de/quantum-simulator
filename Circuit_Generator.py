import os
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.qasm2 import dumps
import math
import random 

def random_circuit_2n(n, p_two_qubit=0.5):
    qc = QuantumCircuit(n)

    for _ in range(2 * n):
        if n > 1 and random.random() < p_two_qubit:
            q1, q2 = random.sample(range(n), 2)
            qc.cx(q1, q2)
        else:
            q = random.randrange(n)
            gate = random.choice(["x", "h", "rz"])
            if gate == "rz":
                qc.rz(random.uniform(0, 2 * math.pi), q)
            elif gate == "h":
                qc.h(q)
            else:
                qc.x(q)

    return qc


def export_random_2n_qasm(n, filename):
    dirpath = "./circuits/gen"
    filepath = os.path.join(dirpath, filename)
    os.makedirs(dirpath, exist_ok=True)

    qc = random_circuit_2n(n)

    # DO NOT decompose random circuits
    qc = transpile(
        qc,
        basis_gates=["rz", "sx", "cx"],
        optimization_level=0
    )

    with open(filepath, "w") as f:
        f.write(dumps(qc))

def generate_qft_qasm(n, filename="qft.qasm"):
    dirpath = "./circuits/gen"
    filepath = os.path.join(dirpath, filename)
    os.makedirs(dirpath, exist_ok=True)

    qc = QuantumCircuit(n)
    qc.append(QFT(n, do_swaps=True), range(n))

    qc = qc.decompose().decompose()

    qc = transpile(
        qc,
        basis_gates=["rz", "sx", "cx"],
        optimization_level=0   
    )

    qasm_str = dumps(qc)

    with open(filepath, "w") as f:
        f.write(qasm_str)

    print(f"Generated IBM-basis QFT QASM with {n} qubits → {filepath}")

def majority(qc, a, b, c):
    qc.cx(c, b)
    qc.cx(c, a)
    qc.ccx(a, b, c)

def unmajority(qc, a, b, c):
    qc.ccx(a, b, c)
    qc.cx(c, a)
    qc.cx(a, b)

def cuccaro_adder(n):
    k = (n - 1) // 2
    qc = QuantumCircuit(n)

    a = list(range(k))
    b = list(range(k, 2*k))
    cin = 2*k

    majority(qc, a[0], b[0], cin)
    for i in range(1, k):
        majority(qc, a[i], b[i], a[i-1])

    for i in reversed(range(1, k)):
        unmajority(qc, a[i], b[i], a[i-1])
    unmajority(qc, a[0], b[0], cin)

    return qc



def export_cuccaro_qasm(n, filename):
    dirpath = "./circuits/gen"
    filepath = os.path.join(dirpath, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    qc = cuccaro_adder(n)
    qc = transpile(qc, basis_gates=["rz", "sx", "cx"], optimization_level=0)
    with open(filepath, "w") as f:
        f.write(dumps(qc))


def draper_adder(n):
    qc = QuantumCircuit(2*n)
    a = list(range(n))
    b = list(range(n, 2*n))

    qc.append(QFT(n, do_swaps=False), b)

    for i in range(n):
        for j in range(n - i):
            qc.cp(2 * math.pi / (2 ** (j + 1)), a[i], b[i + j])

    qc.append(QFT(n, do_swaps=False).inverse(), b)
    return qc


def export_draper_qasm(n, filename):
    dirpath = "./circuits/gen"
    filepath = os.path.join(dirpath, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    qc = draper_adder(n)
    qc = qc.decompose().decompose()
    qc = transpile(qc, basis_gates=["rz", "sx", "cx"], optimization_level=0)
    with open(filepath, "w") as f:
        f.write(dumps(qc))



def cuccaro_adder_bits(m):
    """
    Build a Cuccaro ripple-carry adder that adds two m-bit registers.
    The circuit uses 2*m + 1 qubits: m for register a, m for register b, and 1 carry qubit.
    """
    qc = QuantumCircuit(2*m + 1)

    a = list(range(m))
    b = list(range(m, 2*m))
    cin = 2*m

    # forward majority chain
    majority(qc, a[0], b[0], cin)
    for i in range(1, m):
        majority(qc, a[i], b[i], a[i-1])

    # reverse unmajority chain
    for i in reversed(range(1, m)):
        unmajority(qc, a[i], b[i], a[i-1])
    unmajority(qc, a[0], b[0], cin)

    return qc


def export_cuccaro_bits_qasm(m, filename):
    """Export Cuccaro adder for m-bit operands to QASM.

    Note: total qubits = 2*m + 1 (carry qubit). For m=56 this produces 113 qubits.
    """
    dirpath = "./circuits/gen"
    filepath = os.path.join(dirpath, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    qc = cuccaro_adder_bits(m)
    qc = transpile(qc, basis_gates=["rz", "sx", "cx"], optimization_level=0)
    with open(filepath, "w") as f:
        f.write(dumps(qc))


if __name__ == "__main__":
    generate_qft_qasm(112, "qft_n112.qasm")
    export_draper_qasm(56, "draper_n56_total112.qasm")
    export_cuccaro_bits_qasm(56, "cuccaro_n56_total113.qasm")


#example run circuits
generate_qft_qasm(100,"qft_100.qasm")
export_cuccaro_qasm(100,"cuccaro_100.qasm")
export_draper_qasm(100,"draper_100.qasm")
export_random_2n_qasm(100,"random_100.qasm")


