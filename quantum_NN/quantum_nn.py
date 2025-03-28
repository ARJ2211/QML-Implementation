import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from pennylane.qnn import TorchLayer
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import math

# ======== CONFIG =========
NUM_QUBITS = 16
NUM_CLASSES = 9
LEARNING_RATE = 0.01
BATCH_SIZE = 8
EPOCHS = 50

device = "cpu"

# ======== DATA LOADING =========
def open_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)

def load_dataset(feature_type):
    base = f"q_dataset/{feature_type}_features"
    X_train = torch.stack(open_pkl(os.path.join(base, "mera_features_Train.pkl")))
    y_train = torch.tensor(open_pkl(os.path.join(base, "mera_labels_Train.pkl"))).long()

    X_test = torch.stack(open_pkl(os.path.join(base, "mera_features_Test.pkl")))
    y_test = torch.tensor(open_pkl(os.path.join(base, "mera_labels_Test.pkl"))).long()

    X_val = X_train[2175:]
    y_val = y_train[2175:]
    X_train = X_train[:2175]
    y_train = y_train[:2175]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ======== QCNN DEFINITION =========
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

# -----------------------------------------------------------
# Compute the number of trainable parameters in the QCNN model.
#
# 1. num_blocks:
#    Determines how many conv + pool blocks the network will have.
#    Since each pooling halves the number of qubits, the number of 
#    blocks is log2(NUM_QUBITS) - 2 so that we stop when we have 4 qubits.
#    e.g., for 32 qubits: log2(32) = 5 -> num_blocks = 3
#
# 2. total_conv_weights:
#    For each block i, number of qubit pairs = NUM_QUBITS // 2^i - 1.
#    Each pair has 3 parameters (IsingXX, IsingYY, IsingZZ).
#    So for each block: (num_pairs) × 3
#    e.g., for 32 qubits:
#         Block 0: 15 pairs -> 45 params
#         Block 1: 7 pairs -> 21 params
#         Block 2: 3 pairs -> 9 params
#         Total = 75 conv weights
#
# 3. total_pool_weights:
#    For each block, after pooling, we keep half the qubits.
#    Each of the kept qubits gets a single RY(θ) -> 1 param per kept qubit.
#    e.g., for 32 qubits:
#         Block 0: 16 kept -> 16 weights
#         Block 1: 8 kept -> 8 weights
#         Block 2: 4 kept -> 4 weights
#         Total = 28 pool weights
#
# 4. total_params:
#    Sum of conv + pool weights -> total number of weights to train.
#
# 5. weight_shapes:
#    Passed into TorchLayer to initialize a flat trainable vector 
#    of shape (total_params,), which will be sliced block-wise later.
# -----------------------------------------------------------
num_blocks = int(math.log2(NUM_QUBITS) - 2)
total_conv_weights = sum((NUM_QUBITS // (2 ** i) - 1) * 3 for i in range(num_blocks))
total_pool_weights = sum(NUM_QUBITS // (2 ** (i + 1)) for i in range(num_blocks))
total_params = total_conv_weights + total_pool_weights

weight_shapes = {"weights": (total_params,)}

@qml.qnode(dev, interface="torch")
def qcnn(inputs, weights):
    def state_preparation():
        for i in range(NUM_QUBITS):
            qml.Hadamard(wires=i)
        for i in range(NUM_QUBITS - 1):
            qml.CZ(wires=[i, i + 1])

    def encode_features(x):
        for i in range(NUM_QUBITS):
            qml.RX(x[i], wires=i)
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)

    def conv_layer(qubits, weights):
        for w_idx, i in enumerate(range(0, len(qubits) - 1, 2)):
            qml.IsingXX(
                weights[w_idx][0], wires=[qubits[i], qubits[i+1]]
            )
            qml.IsingYY(
                weights[w_idx][1], wires=[qubits[i], qubits[i+1]]
            )
            qml.IsingZZ(
                weights[w_idx][2], wires=[qubits[i], qubits[i+1]]
            )
            
    def pooling_layer(qubits, theta):
        new_qubits = []
        for i in range(0, len(qubits), 2):
            control, target = qubits[i], qubits[i+1]
            qml.CNOT(wires=[control, target])
            qml.RY(theta[i//2], wires=control)
            new_qubits.append(control)
        return new_qubits

    state_preparation()
    encode_features(inputs)

    qubits = list(range(NUM_QUBITS))
    idx = 0
    for i in range(num_blocks):
        num_pairs = len(qubits) // 2
        conv_weights = weights[idx : idx + num_pairs * 3].reshape((num_pairs, 3))
        idx += num_pairs * 3

        conv_layer(qubits, conv_weights)

        num_kept = len(qubits) // 2
        pool_weights = weights[idx : idx + num_kept]
        idx += num_kept

        qubits = pooling_layer(qubits, pool_weights)

    return [qml.expval(qml.PauliZ(q)) for q in qubits]

torch_qcnn = TorchLayer(qcnn, weight_shapes)

# ======== CLASSIFIER =========
class QuantumPureQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = torch_qcnn
        self.linear = nn.Linear(4, NUM_CLASSES)

    def forward(self, x):
        q_out = [self.qnn(xi) for xi in x]
        q_out = torch.stack(q_out)
        return self.linear(q_out)

# ======== PLOTTING =========
def plot_accuracy(train_acc, val_acc, feature_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"quantum_NN/results_{feature_name}/qcnn_accuracy_plot.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, feature_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"quantum_NN/results_{feature_name}/qcnn_confusion_matrix.png")
    plt.show()

# ======== TRAINING =========
def train(model, train_loader, val_loader, feature_name, epochs):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS if not epochs else epochs):
        model.train()
        correct = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS if not epochs else epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()

        train_acc = correct / len(train_loader.dataset)
        train_accs.append(train_acc)

        model.eval()
        with torch.no_grad():
            correct = 0
            all_preds, all_labels = [], []
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            val_acc = correct / len(val_loader.dataset)
            val_accs.append(val_acc)
        print(
            f"Epoch {epoch+1}/{EPOCHS if not epochs else epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    plot_accuracy(train_accs, val_accs, feature_name)
    plot_confusion_matrix(all_labels, all_preds, feature_name)
    return model

# ======== SUMMARY =========
def print_model_summary(model):
    print("QCNN Model Summary")
    print("=" * 50)
    print("Quantum Layer:")
    print(f"  Name: {type(model.qnn).__name__}")
    total_q_params = sum(p.numel() for p in model.qnn.parameters())
    print(f"  Trainable Quantum Params: {total_q_params}")
    print("-" * 50)
    print("Classical Linear Head:")
    total_c_params = sum(p.numel() for p in model.linear.parameters())
    print(f"  Linear Params: {total_c_params}")
    print("=" * 50)
    print(f"Total Trainable Parameters: {total_q_params + total_c_params}")

# ======== RUN =========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "HOG", "PCA", "PATCH", 
            "FRAC", "FRAC_ENTROPY"
        ],
        required=True,
        help="Choose which feature set to use: HOG, PCA, PATCH or FRAC"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        required=False,
        help="Choose batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        required=False,
        help="Choose epochs to run on"
    )
    args = parser.parse_args()
    os.makedirs(
        f"quantum_NN/results_{args.features}",
        exist_ok=True
    )
    (X_train, y_train), (X_val, y_val), _ = load_dataset(args.features)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = QuantumPureQCNN()
    print_model_summary(model)
    model = train(
        model, 
        train_loader, 
        val_loader, 
        args.features, 
        args.epochs
    )
    torch.save(
        model.state_dict(), f"quantum_NN/results_{args.feature}/qcnn_model_weights.pth"
    )
    print(
        f"Model saved to quantum_NN/results_{args.feature}/qcnn_model_weights.pth"
    )
