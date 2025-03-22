import pickle as pkl
from mera_processing import QuantumISICDataset

with open(r'quantum_isic_dataset.pkl', 'rb') as f:
    data = pkl.load(f)

print(data)