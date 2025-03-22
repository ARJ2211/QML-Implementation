import pennylane as qml
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import pickle as pkl
from tqdm import tqdm
import os
import random

random.seed(42)
np.random.seed(seed=42)
global pca_model 

# Image transformer
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),  # PCA must train on 256x256 images
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Define quantum device
num_qubits = 8  # Set number of qubits
dev = qml.device("default.qubit", wires=num_qubits)

# Define a quantum block for MERA
def quantum_block(weights, wires):
    """A simple two-qubit unitary block"""
    # Removes short term entanglement, if all qubits are fully entangled,
    # computations become exponentially complex. This is like pooling
    qml.Hadamard(wires=wires[0])  # Superposition so information is spread
    qml.CNOT(wires=wires)  # Entanglement of the two qubits to preserve correlations

    # These act like trainable transformations like filters in CNN's
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

@qml.qnode(dev)
def mera_circuit(inputs):
    """Quantum MERA for feature extraction"""
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")

    num_layers = 15 # Reduced layers $$2log_2{N}+\alpha$$
    num_params_per_block = 2
    num_blocks = num_qubits // 2

    # Trainable weights for rY gates (rotation parameters)
    template_weights = np.random.randn(num_layers, num_blocks, num_params_per_block)

    for layer in range(num_layers):
        # Iterate over the number of layers
        for i in range(0, num_qubits - 1, 2):
            # Applies entanglement + rY rotations
            quantum_block(template_weights[layer][i // 2], [i, i + 1])

    # Apply pauliZ measurement where $$\langle Z_i\rangle = \langle\Psi\vert Z_i\vert\Psi\rangle$$
    return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

def image_to_quantum(image, pca_model, draw=False):
    """Encodes classical image data into a quantum state using PCA + Angle Encoding."""
    if pca_model is None:
        raise ValueError("PCA model is None! Ensure PCA is trained before calling image_to_quantum.")

    if isinstance(image, torch.Tensor):
        image = image.numpy()
        image = np.squeeze(image)
        image = Image.fromarray((image * 255).astype('uint8'))

    transform = TRANSFORM

    image = transform(image)
    image = image.squeeze(0)  # Remove channel dim

    # Flatten image (8x8 → 64 pixels)
    flat_image = image.flatten().reshape(1, -1)

    # Apply PCA transformation
    reduced_features = pca_model.transform(flat_image)  # Use trained PCA model
    reduced_features = reduced_features.flatten()

    # Normalize PCA output to fit within [0, π] for encoding
    reduced_features = np.interp(reduced_features, (reduced_features.min(), reduced_features.max()), (0, np.pi))

    quantum_output = np.array(mera_circuit(reduced_features))
    if draw:
        drawer = qml.draw(mera_circuit)
        print(drawer(reduced_features))
        draw -= 1

    return torch.tensor(quantum_output, dtype=torch.float32)

# Train PCA model on dataset images
def train_pca(dataset, n_components=num_qubits):
    """Trains PCA on a sample of dataset images."""
    all_images = [dataset[i][0] for i in range((len(dataset)))]  # Extract only image data

    all_images = np.array(all_images)
    
    print(f"PCA Training Data Shape: {all_images.shape}")  # Debugging
    pca = PCA(n_components=n_components)
    pca.fit(all_images)  # Train PCA
    return pca

class RawISICDataset(Dataset):
    """Dataset to load raw images for PCA training without quantum processing."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")
        
        transform = TRANSFORM

        image = transform(image)

        return image.squeeze(0).flatten().numpy(), label

# Custom dataset loader
class QuantumISICDataset(Dataset):
    def __init__(self, root_dir, pca_model):
        self.root_dir = root_dir
        self.pca_model = pca_model  # Store PCA model
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
        
        self.image_paths, self.labels = self.__shuffle__()

    def __shuffle__(self):
        shuff = list(zip(self.image_paths, self.labels))
        random.shuffle(shuff)
        a, b = zip(*shuff)
        return a, b

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")

        quantum_features = image_to_quantum(image, self.pca_model)

        return quantum_features, label


def create_features(pca_model, type:str = 'Train'):
    # **Step 1: Create `RawISICDataset` to train PCA**
    if type == "Train":
        raw_dataset = RawISICDataset(root_dir="datasets/ISIC/Original/Train")

        # **Step 2: Train PCA on raw images**
        pca_model = train_pca(raw_dataset, num_qubits)
        print("PCA training complete.")
    
    dataset = QuantumISICDataset(root_dir=f"datasets/ISIC/Original/{type}", pca_model=pca_model)

    # Sample quantum feature output
    sample_features, sample_label = dataset[0]
    print("Quantum Feature Vector:\n", sample_features)
    print("Label:", sample_label)

    # **Step 3: Create `QuantumISICDataset` with trained PCA model**
    all_features, all_labels = [], []

    for i in tqdm(range(dataset.__len__()), desc='Feature Extraction'):
        feature, label = dataset[i]
        all_features.append(feature)
        all_labels.append(label)

    with open(f'mera_features_{type}.pkl', 'wb') as f:
        pkl.dump(all_features, f)
        print('Dumped mera features')
    f.close()

    with open(f'mera_labels_{type}.pkl', 'wb') as f:
        pkl.dump(all_labels, f)
        print('Dumped mera labels')
    f.close()

    inputs = np.random.rand(num_qubits)
    # Print the circuit
    drawer = qml.draw(mera_circuit)
    print(drawer(inputs))

    return pca_model



if __name__ == '__main__':
    model_pca = create_features('Train')
    create_features(model_pca, 'Test')
