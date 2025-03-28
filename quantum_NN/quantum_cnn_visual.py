import matplotlib.pyplot as plt
import warnings
import pennylane as qml
import numpy as np

warnings.filterwarnings("ignore")

NUM_QUBITS = 16
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

# ---- State Preparation Layer (H + CZ for cluster state) ----
def state_preparation():
    # Apply haramard gate on all 16 qubits
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)

    # Apply Controlled-Z gate on the two
    # subsequent qubits
    for i in range(NUM_QUBITS - 1):
        qml.CZ(wires=[i, i + 1])

# ---- Feature Encoding Layer ----
def encode_features(x):
    # Encode each feature with RX, RY, RZ rotation
    for i in range(NUM_QUBITS):
        qml.RX(x[i], wires=i)
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)

# ---- Convolution Layer (Non-trainable for now) ----
def conv_layer():
    """
    Introduces entanglement between qubits
    Captures Correlations across both qubits
    Acts like a quantum Conv filter
    """
    for i in range(0, NUM_QUBITS - 1, 2):
        qml.IsingXX(np.pi/4, wires=[i, i+1])
        qml.IsingYY(np.pi/4, wires=[i, i+1])
        qml.IsingZZ(np.pi/4, wires=[i, i+1])

# ---- Pooling Layer (reduce qubits by half) ----
def pooling_layer(qubits):
    """
    Reduces the number of qubits by doing CNOT and RY
    and only keeping the control qubits
    """
    new_qubits = []
    for i in range(0, len(qubits), 2):
        control, target = qubits[i], qubits[i+1]
        qml.CNOT(wires=[control, target])
        qml.RY(np.pi/4, wires=control)
        new_qubits.append(control)  # Keep only control
    return new_qubits

# ---- Full QCNN ----
@qml.qnode(dev)
def qcnn_forward(x):
    state_preparation()
    encode_features(x)

    qubits = list(range(NUM_QUBITS))  # 16 -> 8 -> 4 -> 2 -> 1
    print('QUBITS: ', qubits)
    while len(qubits) > 4:
        conv_layer()
        qubits = pooling_layer(qubits)
        print('QUBITS: ', qubits)

    return [
        qml.expval(qml.PauliZ(q)) for q in qubits
    ]  # Final 4 qubit output

dummy_input = np.random.uniform(
    -np.pi, 
    np.pi, 
    size=(NUM_QUBITS,)
)
out = qcnn_forward(dummy_input)
print("Output PauliZ on Qubit 0:", out)
qml.draw_mpl(qcnn_forward)(dummy_input)
plt.savefig(r'quantum_NN/QCNN_Visualization.jpg')