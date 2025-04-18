import torch
import torch.nn as nn
import torch.optim as optim
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

# ========== CONFIG ==========
NUM_QUBITS = 16
NUM_LAYERS = 29
NUM_CLASSES = 9
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.01

# ========== DATA LOADERS ==========
def open_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def load_dataset(feature_type):
    base_path = f"q_dataset/{feature_type}_features"
    
    X_train = torch.stack(open_pkl(os.path.join(base_path, "mera_features_Train.pkl")))
    y_train = torch.tensor(open_pkl(os.path.join(base_path, "mera_labels_Train.pkl"))).long()

    X_test = torch.stack(open_pkl(os.path.join(base_path, "mera_features_Test.pkl")))
    y_test = torch.tensor(open_pkl(os.path.join(base_path, "mera_labels_Test.pkl"))).long()

    X_val = X_train[2175:]
    y_val = y_train[2175:]
    X_train = X_train[:2175]
    y_train = y_train[:2175]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ========== QUANTUM BLOCK ==========
def mera_block(weights, wires):
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=wires)
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

# ========== QNODE ==========
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface='torch')
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
    qml.templates.MERA(
        wires=range(NUM_QUBITS),
        n_block_wires=2,
        block=mera_block,
        n_params_block=2,
        template_weights=weights
    )
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

weight_shapes = {"weights": (NUM_LAYERS, NUM_QUBITS // 2, 2)}
q_layer = TorchLayer(qnode, weight_shapes)

# ========== MODEL ==========
class HybridModel(nn.Module):
    def __init__(self, feature_type):
        super().__init__()
        self.feature_type = feature_type.upper()
        self.qnn = q_layer

        # Input size based on feature type
        if self.feature_type in ["HOG", "PCA", "FRAC"]:
            self.input_dim = NUM_QUBITS*2
        elif self.feature_type == "PATCH":
            self.input_dim = NUM_QUBITS * NUM_QUBITS * 2  # 16 patches -> each 16-dim output -> total 256
        else:
            raise ValueError("Unsupported feature type")

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        q_out = []

        if self.feature_type in ["HOG", "PCA", "FRAC"]:
            # x.shape = (B, 16)
            for xi in x:
                q_out.append(self.qnn(xi))  # Each xi → (16,)
        elif self.feature_type == "PATCH":
            # import pdb; pdb.set_trace()
            # x.shape = (B, 16, 16)
            x = x.view(-1, 16, 16)
            for xi in x:  # xi: (16, 16)
                patch_outputs = [self.qnn(patch).flatten() for patch in xi]  # 16 × (16,)
                merged = torch.cat(patch_outputs, dim=0)  # → (256,)
                q_out.append(merged)

        q_out = torch.stack(q_out)  # Final shape: (B, 256)
        return self.classifier(q_out)


# ========== SCHEDULER ==========
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# ========== TRAINING ==========
def train_model(
        model,
        train_loader,
        val_loader,
        device,
        feature_name,
        epochs
    ):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer)
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS if not epochs else epochs):
        model.train()
        correct = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS if not epochs else epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
        train_acc = correct / len(train_loader.dataset)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
        val_acc = correct / len(val_loader.dataset)
        val_accs.append(val_acc)

        scheduler.step()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS if not epochs else epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    torch.save(model.state_dict(), f"quantum_hybrid_NN/results_{feature_name}/hybrid_model_weights.pth")
    print(f"Model saved to quantum_hybrid_NN/results_{feature_name}/hybrid_model_weights.pth")
    return train_accs, val_accs

# ========== EVALUATION ==========
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return all_preds, all_labels

# =========== SUMMARY ============
def print_model_summary(model, input_shape):
    print("Hybrid Quantum-Classical Model Summary")
    print("="*60)
    print("Quantum Layer:")
    print(f"  Name: {type(model.qnn).__name__}")
    print(f"  Input Shape: {input_shape}")
    print(f"  Output Shape: ({NUM_QUBITS * 2},)")
    total_q_params = sum(p.numel() for p in model.qnn.parameters())
    print(f"  Trainable Params: {total_q_params}")
    print("-"*60)
    print("Classical Classifier:")
    total_c_params = 0
    for idx, layer in enumerate(model.classifier):
        name = type(layer).__name__
        params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total_c_params += params
        print(f"  [{idx}] {name:<15} -> Params: {params}")
    print("="*60)
    print(f"Total Parameters: {total_q_params + total_c_params}")
    print(f"Trainable: {total_q_params + total_c_params}")
    print("="*60)

# ========== PLOTTING ==========
def plot_accuracy(train_acc, val_acc, feature_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"quantum_hybrid_NN/results_{feature_name}/hybrid_accuracy_plot.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, feature_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"quantum_hybrid_NN/results_{feature_name}/hybrid_confusion_matrix.png")
    plt.show()

# ========== RUN ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        choices=["HOG", "PCA", "PATCH", "FRAC"],
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
        f"quantum_hybrid_NN/results_{args.features}",
        exist_ok=True
    )
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Device: {device}")
    print(f"Using features from: {args.features}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.features)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE if not args.batch else args.batch, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=BATCH_SIZE if not args.batch else args.batch,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE if not args.batch else args.batch,
        shuffle=True
    )
    # Infer input size based on feature type
    model = HybridModel(feature_type=args.features)
    print_model_summary(model, input_shape=(NUM_QUBITS,))
    train_acc, val_acc = train_model(model, train_loader, val_loader, device, args.features, args.epochs)
    plot_accuracy(train_acc, val_acc, args.features)

    y_pred, y_true = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, args.features)
    print("Evaluation complete.")

