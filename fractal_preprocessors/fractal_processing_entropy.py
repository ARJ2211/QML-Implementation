import os
import numpy as np
import torch
import pickle as pkl
import random
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.filters import sobel, threshold_otsu
from skimage.transform import resize
from skimage.util import view_as_blocks

warnings.filterwarnings('ignore')

# ========== CONFIGURATIONS =============
random.seed(42)
np.random.seed(42)

FEATURE_PATH = 'q_dataset/FRAC_ENTROPY_features/'
os.makedirs(FEATURE_PATH, exist_ok=True)

NUM_QUBITS = 16  # 1 slope + 7 entropies + 8 sobel
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# =========== ENTROPY MAP ============
def entropy_map(image, grid_size=(4, 4)):
    """
    Divides image into grids and computes entropy for each grid.
    Returns a flattened entropy map vector.
    """
    h, w = image.shape
    gh, gw = grid_size
    block_h, block_w = h // gh, w // gw
    entropies = []
    for i in range(gh):
        for j in range(gw):
            block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            hist, _ = np.histogram(block, bins=16, range=(0, 1), density=True)
            hist = hist + 1e-9  # prevent log(0)
            entropy = -np.sum(hist * np.log(hist))
            entropies.append(entropy)
    return np.array(entropies)

# =========== FRACTAL FEATURE EXTRACTOR ============
def extract_fractal_features(image, scales=[4, 8, 16, 32, 64, 128, 256]):
    """
    Extracts 8 fractal features from binary image: slope + 7 entropies.
    """
    log_scales, log_counts, entropy_features = [], [], []

    for scale in scales:
        block_shape = (256 // scale, 256 // scale)
        blocks = view_as_blocks(image, block_shape)

        # Reshape to (num_blocks, h, w)
        non_empty_blocks = blocks.reshape(-1, *block_shape)
        non_empty = np.sum(np.any(non_empty_blocks > 0, axis=(1, 2)))

        log_scales.append(np.log(1.0 / scale))
        log_counts.append(np.log(non_empty + 1))  # +1 to avoid log(0)

        block_sums = blocks.sum(axis=(-1, -2)).flatten()
        total = np.sum(block_sums)
        probs = block_sums / total if total > 0 else np.zeros_like(block_sums)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        entropy_features.append(entropy)

    # Linear regression to get slope (Box-Counting dimension approx)
    try:
        A = np.vstack([log_scales, np.ones(len(log_scales))]).T
        slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
    except Exception:
        slope = 0.0

    return [slope] + entropy_features  # 8 features

def image_to_fractal_features(image, stats):
    """
    Applies fractal feature extraction on original and Sobel image.
    Adds entropy map values. Standardizes with precomputed stats.
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze()
        image = (image * 255).astype(np.uint8)
    else:
        image = np.array(image.convert("L"))

    image = resize(image, (256, 256), anti_aliasing=True)

    # Adaptive thresholding
    thresh = threshold_otsu(image)
    image_bin = (image > thresh).astype(np.float32)

    sobel_img = sobel(image)
    sobel_thresh = threshold_otsu(sobel_img)
    sobel_bin = (sobel_img > sobel_thresh).astype(np.float32)

    features_orig = extract_fractal_features(image_bin)
    features_sobel = extract_fractal_features(sobel_bin)

    # Append 2D entropy maps (e.g., 4x4 = 16D)
    emap = entropy_map(image, grid_size=(4, 4))
    features = np.concatenate([features_orig, features_sobel, emap])  # 8+8+16 = 32D

    standardized = (features - stats["mean"]) / (stats["std"] + 1e-8)
    return torch.tensor(standardized, dtype=torch.float32)

# ========== DATASET ==============
class RawFractalISICDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir): continue
            for img in os.listdir(class_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img))
                    self.labels.append(label)

        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        return image, label

# ============ MAIN DRIVER ==============
def create_fractal_features(type="Train"):
    dataset = RawFractalISICDataset(root_dir=f"datasets/ISIC/Original/{type}")

    raw_features, labels = [], []

    print(f"[{type}] Extracting raw fractal features...")
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        img_np = np.array(img.convert("L"))
        img_np = resize(img_np, (256, 256), anti_aliasing=True)

        thresh_img = (img_np > threshold_otsu(img_np)).astype(np.float32)
        sobel_img = sobel(img_np)
        thresh_sobel = (sobel_img > threshold_otsu(sobel_img)).astype(np.float32)

        f1 = extract_fractal_features(thresh_img)
        f2 = extract_fractal_features(thresh_sobel)
        emap = entropy_map(img_np, grid_size=(4, 4))

        combined = np.concatenate([f1, f2, emap])
        raw_features.append(combined)
        labels.append(label)

    raw_features = np.array(raw_features)

    if type == "Train":
        mean = raw_features.mean(axis=0)
        std = raw_features.std(axis=0)
        stats = {"mean": mean, "std": std}
        with open(os.path.join(FEATURE_PATH, "global_stats.pkl"), 'wb') as f:
            pkl.dump(stats, f)
        print("Saved per-feature mean/std.")
    else:
        with open(os.path.join(FEATURE_PATH, "global_stats.pkl"), 'rb') as f:
            stats = pkl.load(f)
        print("Loaded per-feature mean/std.")

    # Normalize
    norm_features = [
        image_to_fractal_features(dataset[i][0], stats)
        for i in range(len(dataset))
    ]

    # Save features and labels
    with open(os.path.join(FEATURE_PATH, f"mera_features_{type}.pkl"), "wb") as f:
        pkl.dump(norm_features, f)
        print(f"Dumped mera_features_{type}.pkl")

    with open(os.path.join(FEATURE_PATH, f"mera_labels_{type}.pkl"), "wb") as f:
        pkl.dump(labels, f)
        print(f"Dumped mera_labels_{type}.pkl")

    # Visualization
    plt.figure(figsize=(10, 6))
    for i in range(min(5, len(norm_features))):
        plt.bar(
            np.arange(len(norm_features[i])) + i * 0.15,
            norm_features[i], width=0.15,
            label=f'Sample: {i} Label: {labels[i]}'
        )
    plt.xticks(np.arange(len(norm_features[0])), [f"F{i+1}" for i in range(len(norm_features[0]))])
    plt.ylabel("Normalized Feature Value")
    plt.title(f"Fractal Quantum Features - {type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_PATH, f"fractal_feature_vectors_{type}.png"))
    print(f"Saved visualization: fractal_feature_vectors_{type}.png")

    return norm_features, labels

# ============ EXECUTION ============
if __name__ == '__main__':
    create_fractal_features("Train")
    create_fractal_features("Test")
