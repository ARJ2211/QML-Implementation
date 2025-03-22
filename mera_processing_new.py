import os
import pennylane as qml
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import pickle as pkl
from tqdm import tqdm
from skimage.feature import hog


# === CONFIGURATIONS ===
NUM_QUBITS = 16
IMAGE_SIZE = 64  
NUM_PCA_COMPONENTS = NUM_QUBITS  
NUM_LAYERS = 6  
random.seed(42)
np.random.seed(seed=42)
global pca_model 


# Trainable MERA weights (frozen for now)
template_weights = np.random.randn(NUM_LAYERS, NUM_QUBITS // 2, 2)

# Ensure PCA directory exists
os.makedirs("q_dataset", exist_ok=True)

# === IMAGE TRANSFORM ===
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# === QUANTUM DEVICE ===
dev = qml.device("default.qubit", wires=NUM_QUBITS)

# === MERA BLOCK ===
def quantum_block(weights, wires):
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=wires)
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

# === MERA QNODE ===
@qml.qnode(dev)
def mera_circuit(inputs):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
    for layer in range(NUM_LAYERS):
        for i in range(0, NUM_QUBITS - 1, 2):
            quantum_block(template_weights[layer][i // 2], [i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# === HOG FEATURE EXTRACTION ===
def hog_preprocess(image):
    """ Extracts HOG features safely from a 2D image. """
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise ValueError("Input image must be at least 8x8 for HOG.")

    return hog(image, orientations=8, pixels_per_cell=(8, 8), 
               cells_per_block=(1, 1), visualize=False, feature_vector=True)

# === PCA TRAINING ===
def train_pca(dataset):
    """ Trains PCA on HOG features from dataset images. """
    hog_features = np.array([hog_preprocess(dataset[i][0]) for i in range(len(dataset))])

    print(f"PCA training data shape: {hog_features.shape}")
    pca = PCA(n_components=NUM_PCA_COMPONENTS)
    pca.fit(hog_features)
    return pca

# === IMAGE -> QUANTUM PIPELINE ===
def image_to_quantum(image, pca_model):
    """ Process image: HOG → PCA → Quantum Circuit """
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        image = np.squeeze(image)
        image = Image.fromarray((image * 255).astype('uint8'))

    image = TRANSFORM(image).squeeze(0)

    # Corrected: Apply HOG before PCA
    hog_features = hog_preprocess(image)
    reduced = pca_model.transform([hog_features]).flatten()
    normalized = np.interp(reduced, (reduced.min(), reduced.max()), (0, np.pi))

    return torch.tensor(mera_circuit(normalized), dtype=torch.float32)

# === DATASET LOADER CLASSES ===
class RawISICDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths, self.labels = [], []
        self.classes = sorted(os.listdir(root_dir))
        for label, cls in enumerate(self.classes):
            for img in os.listdir(os.path.join(root_dir, cls)):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, cls, img))
                    self.labels.append(label)
    def __len__(self): 
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        return TRANSFORM(img).squeeze(0), self.labels[idx]  # 2D image (64x64)

class QuantumISICDataset(Dataset):
    def __init__(self, root_dir, pca_model):
        self.pca_model = pca_model
        self.image_paths, self.labels = [], []
        self.classes = sorted(os.listdir(root_dir))
        for label, cls in enumerate(self.classes):
            for img in os.listdir(os.path.join(root_dir, cls)):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, cls, img))
                    self.labels.append(label)
        zipped = list(zip(self.image_paths, self.labels))
        random.shuffle(zipped)
        self.image_paths, self.labels = zip(*zipped)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        return image_to_quantum(img, self.pca_model), self.labels[idx]

# === FEATURE EXTRACTION PIPELINE ===
def create_features(pca_model=None, split='Train'):
    if split == 'Train':
        raw_dataset = RawISICDataset(f"datasets/ISIC/Original/{split}")
        pca_model = train_pca(raw_dataset)

    quantum_dataset = QuantumISICDataset(f"datasets/ISIC/Original/{split}", pca_model)

    features, labels = [], []
    for i in tqdm(range(len(quantum_dataset)), desc=f"Extracting {split} features"):
        feat, label = quantum_dataset[i]
        features.append(feat)
        labels.append(label)

    # Save extracted features
    with open(f'q_dataset/mera_features_{split}.pkl', 'wb') as f: pkl.dump(features, f)
    with open(f'q_dataset/mera_labels_{split}.pkl', 'wb') as f: pkl.dump(labels, f)

    return pca_model

# === MAIN ===
if __name__ == "__main__":
    pca = create_features(None, 'Train')
    create_features(pca, 'Test')
