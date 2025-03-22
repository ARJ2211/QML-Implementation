import tensorflow as tf
import numpy as np
import torch
import keras
import pickle as pkl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

# Create results directory
os.makedirs("results", exist_ok=True)

# Function to plot and save accuracy/loss graphs
def plot_graph(history):
    train_acc = [i * 100 for i in history.history['accuracy']]
    val_acc = [i * 100 for i in history.history.get('val_accuracy', [])]

    plt.figure(figsize=(14, 7))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy', marker='o')
    if val_acc:
        plt.plot(val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()

    # Save plot
    plt.savefig('results/accuracy_loss_plot.png')
    plt.show()


# One-hot encoder
def to_one_hot(y, num_class):
    one_hot = np.zeros((y.shape[0], num_class))
    for i in range(y.shape[0]):
        one_hot[i][y[i][0]] = 1
    return one_hot


# Save confusion matrix
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("results/confusion_matrix.png")
    plt.show()


# Load pickled data
def open_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


# Define the classical model
def build_model_nn(X_train):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)))
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    # model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
    # model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(9, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# === MAIN ===
if __name__ == "__main__":
    # Load quantum-inspired features
    X_train_raw = open_pkl('q_dataset/mera_features_Train.pkl')
    y_train_raw = open_pkl('q_dataset/mera_labels_Train.pkl')
    X_test_raw = open_pkl('q_dataset/mera_features_Test.pkl')
    y_test_raw = open_pkl('q_dataset/mera_labels_Test.pkl')

    # Convert to NumPy
    X_train = torch.stack(X_train_raw).numpy()
    y_train = np.array(y_train_raw).reshape(-1, 1)
    X_test = torch.stack(X_test_raw).numpy()
    y_test = np.array(y_test_raw).reshape(-1, 1)

    # Train/val split
    X_train, X_val = X_train[:2175], X_train[2175:]
    y_train, y_val = y_train[:2175], y_train[2175:]

    # One-hot encode
    y_train = to_one_hot(y_train, 9)
    y_val = to_one_hot(y_val, 9)
    y_test = to_one_hot(y_test, 9)

    # Print data info
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    # Train model
    model = build_model_nn(X_train)
    with open("results/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=8,
                        epochs=100,
                        verbose=1)

    # Save training history
    with open("results/training_history.pkl", "wb") as f:
        pkl.dump(history.history, f)

    # Save and show plots
    plot_graph(history)

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    plot_confusion(y_true, y_pred)
