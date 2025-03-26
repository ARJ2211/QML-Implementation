import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pennylane as qml
import pickle as pkl
from tqdm import tqdm
import random

# ========== CONFIGURATION ==========
PATCHES = 16
PATCH_SIZE = 16
IMAGE_SIZE = PATCH_SIZE * int(PATCHES**0.5)  # 256
NUM_QUBITS = PATCH_SIZE  # 16
NUM_LAYERS = qml.templates.MERA.get_n_blocks(
    wires=range(NUM_QUBITS), n_block_wires=2
)
random.seed(42)
np.random.seed(42)
os.makedirs("q_dataset/PATCH_features", exist_ok=True)

# ========== IMAGE TRANSFORM ==========
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# ========== QUANTUM SETUP ==========
template_weights = np.random.randn(NUM_LAYERS, 2)
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

def mera_block(weights, wires):
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=wires)
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

@qml.qnode(dev)
def mera_patch_circuit(inputs):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
    qml.templates.MERA(
        wires=range(NUM_QUBITS),
        n_block_wires=2,
        block=mera_block,
        n_params_block=2,
        template_weights=template_weights
    )
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

def image_to_quantum_patches(image):
    """Divide 256x256 image into 16 patches -> downsample each patch to 16 values -> MERA encode."""
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    if image.ndim == 3:
        image = np.squeeze(image, axis=0)
    elif image.ndim != 2:
        raise ValueError(f"Invalid image shape: {image.shape}")

    # Divide into 16 patches (4×4 grid of 64×64 patches)
    patch_features = []
    patch_size = IMAGE_SIZE // 4  # 64
    for i in range(0, IMAGE_SIZE, patch_size):
        for j in range(0, IMAGE_SIZE, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]

            # Downsample patch to 16 values by averaging 16x16 grids inside the patch
            downsampled = []
            sub_patch_size = patch_size // 4  # 16
            for pi in range(0, patch_size, sub_patch_size):
                for pj in range(0, patch_size, sub_patch_size):
                    sub_patch = patch[pi:pi+sub_patch_size, pj:pj+sub_patch_size]
                    downsampled.append(np.mean(sub_patch))

            assert len(downsampled) == NUM_QUBITS, f"Expected 16 values per patch, got {len(downsampled)}"

            # Normalize to [0, π] for angle encoding
            encoded = np.interp(downsampled, (np.min(downsampled), np.max(downsampled)), (0, np.pi))
            patch_feature = mera_patch_circuit(encoded)
            patch_features.append(patch_feature)
    output = torch.tensor(np.concatenate(patch_features), dtype=torch.float32)
    # print(len(output))
    return output

# ========== DATASETS ==========
class RawPatchDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths, self.labels = [], []
        self.classes = sorted(os.listdir(root_dir))
        for label, cls in enumerate(self.classes):
            for img in os.listdir(os.path.join(root_dir, cls)):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root_dir, cls, img))
                    self.labels.append(label)
        self.image_paths, self.labels = self._shuffle()

    def _shuffle(self):
        zipped = list(zip(self.image_paths, self.labels))
        random.shuffle(zipped)
        return zip(*zipped)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = TRANSFORM(img)
        return img.squeeze(0), self.labels[idx]

class QuantumPatchDataset(Dataset):
    def __init__(self, root_dir):
        self.raw_dataset = RawPatchDataset(root_dir)

    def __len__(self): return len(self.raw_dataset)

    def __getitem__(self, idx):
        img, label = self.raw_dataset[idx]
        quantum_feature = image_to_quantum_patches(img)
        return quantum_feature, label

# ========== MAIN PIPELINE ==========
def create_patch_features(split="Train"):
    dataset = QuantumPatchDataset(f"datasets/ISIC/Original/{split}")
    features, labels = [], []

    for i in tqdm(range(len(dataset)), desc=f"Extracting {split} Patch Features"):
        feat, label = dataset[i]
        features.append(feat)
        labels.append(label)

    with open(f"q_dataset/PATCH_features/mera_features_{split}.pkl", "wb") as f:
        pkl.dump(features, f)
    with open(f"q_dataset/PATCH_features/mera_labels_{split}.pkl", "wb") as f:
        pkl.dump(labels, f)
    print(f"Saved PATCH-based features for {split}")

# ========== RUN ==========
if __name__ == "__main__":
    create_patch_features("Train")
    create_patch_features("Test")
