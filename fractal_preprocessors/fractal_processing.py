import os
import numpy as np
import torch
import pickle as pkl
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.util import view_as_blocks
from skimage.transform import resize

# ---------------------- Configs ----------------------

random.seed(42)
np.random.seed(42)

os.makedirs(r'q_dataset/FRAC_features/', exist_ok=True)

NUM_QUBITS = 8  # 1 fractal slope + 7 entropy-like box-counting values

# Image transformer to ensure 256x256 grayscale input
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# ---------------------- Feature Extractor ----------------------

def extract_fractal_features(image, scales=[2, 4, 8, 16, 32, 64, 128]):
    """
    Extracts fractal features using box-counting and entropy-like spatial complexity.
    Returns: [fractal_dim_slope, entropy_1, entropy_2, ...]
    """
    image = resize(image, (256, 256), anti_aliasing=True)
    image = (image > 0.1).astype(np.uint8)  # binarize image

    log_scales, log_counts, entropy_features = [], [], []

    for scale in scales:
        block_shape = (256 // scale, 256 // scale)
        blocks = view_as_blocks(image, block_shape)

        # Count non-empty boxes (box with at least 1 active pixel)
        non_empty_blocks = np.sum(np.any(np.any(blocks, axis=-1), axis=-1))
        log_scales.append(np.log(1.0 / scale))
        log_counts.append(np.log(non_empty_blocks))

        # Entropy-like spatial distribution measure
        block_sums = blocks.sum(axis=(-1, -2))
        total_mass = np.sum(block_sums)
        probs = block_sums / total_mass if total_mass > 0 else np.zeros_like(block_sums)
        entropy = -np.sum(probs * np.log(probs + 1e-9))  # Add small epsilon to avoid log(0)
        entropy_features.append(entropy)

    # Fit line: log(N(L)) vs log(1/L) → slope = fractal dimension
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]

    return [slope] + entropy_features

def image_to_fractal_features(image):
    """
    Given a PIL or Tensor image, return a quantum-normalized (0 to π) fractal feature vector.
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze()
        image = Image.fromarray((image * 255).astype('uint8'))

    image = TRANSFORM(image).squeeze(0).numpy()  # Apply transform and convert to numpy array

    features = extract_fractal_features(image)

    # Normalize features to [0, π] for use as quantum gate angles

    min_val = np.min(features)
    max_val = np.max(features)
    norm_features = np.interp(features, (min_val, max_val), (0, np.pi))

    return torch.tensor(norm_features, dtype=torch.float32)

# ---------------------- Dataset Classes ----------------------

class RawFractalISICDataset(Dataset):
    """
    Raw loader for ISIC images, used before extracting features.
    """
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_path, img_name))
                    self.labels.append(label)

        # Shuffle dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        return img, label

class QuantumFractalDataset(Dataset):
    """
    Dataset returning quantum-ready fractal feature vectors and their labels.
    """
    def __init__(self, root_dir):
        self.dataset = RawFractalISICDataset(root_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        fractal_features = image_to_fractal_features(image)
        return fractal_features, label

# ---------------------- Driver Function ----------------------

def create_fractal_features(type="Train"):
    """
    Generates and saves fractal quantum feature vectors for all ISIC images.
    Type: 'Train' or 'Test'
    """
    dataset = QuantumFractalDataset(root_dir=f"datasets/ISIC/Original/{type}")
    all_features, all_labels = [], []

    for i in tqdm(range(len(dataset)), desc=f"Fractal Feature Extraction [{type}]"):
        features, label = dataset[i]
        all_features.append(features)
        all_labels.append(label)

    with open(f'q_dataset/FRAC_features/mera_features_{type}.pkl', 'wb') as f:
        pkl.dump(all_features, f)
        print(f"Dumped fractal_features_{type}.pkl")

    with open(f'q_dataset/FRAC_features/mera_labels_{type}.pkl', 'wb') as f:
        pkl.dump(all_labels, f)
        print(f"Dumped fractal_labels_{type}.pkl")

    # Print sample for verification
    print(f"\nSample quantum fractal feature vector:\n{all_features[0]}")
    print("Label:", all_labels[0])

    # Visualize the first 5 feature vectors as bar plots
    plt.figure(figsize=(10, 6))
    for i in range(min(5, len(all_features))):
        plt.bar(
            np.arange(NUM_QUBITS) + i * 0.15,
            all_features[i], width=0.15,
            label=f'Sample {i}'
        )
    plt.xticks(
        np.arange(NUM_QUBITS),
        [f'Entropy {i+1}' for i in range(NUM_QUBITS)]
    )
    plt.ylabel('Rotation Angle (0 to π)')
    plt.title(f'Fractal Quantum Features - {type}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'q_dataset/FRAC_features/fractal_feature_vectors_{type}.png')
    print(f"Saved visualization: fractal_feature_vectors_{type}.png")

    return dataset

# ---------------------- Entry ----------------------

if __name__ == '__main__':
    create_fractal_features("Train")
    create_fractal_features("Test")
