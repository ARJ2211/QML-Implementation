from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl
import torch

BASES = [
    r'q_dataset/FRAC_ENTROPY_features/',
    r'q_dataset/FRAC_features/',
    r'q_dataset/HOG_features/',
    r'q_dataset/PATCH_features/',
    r'q_dataset/PCA_features/'
]
for BASE in BASES:
    # Load features and labels
    with open(f"{BASE}mera_features_Train.pkl", "rb") as f:
        X_train = np.array(pkl.load(f))

    with open(f"{BASE}mera_labels_Train.pkl", "rb") as f:
        y_train = np.array(pkl.load(f))

    with open(f"{BASE}mera_features_Test.pkl", "rb") as f:
        X_test = np.array(pkl.load(f))

    with open(f"{BASE}mera_labels_Test.pkl", "rb") as f:
        y_test = np.array(pkl.load(f))

    # Flatten Torch tensors if needed
    if isinstance(X_train[0], torch.Tensor):
        X_train = np.array([x.numpy() for x in X_train])
        X_test = np.array([x.numpy() for x in X_test])

    # Fit Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy @ {BASE.split('/')[-2]}: {acc * 100:.2f}%")
