# main.py

import torch.optim as optim
import torch.nn as nn
import torch
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 🌟 New: Import OS module for file operations

# Import configurations and modules
from config import FILE_PATH, STEP_LENGTH, KNN_K, EPOCHS, LEARNING_RATE, print_memory_usage
from features import load_and_segment_data, feature_reduction_pca
from graph_model import build_graph_knn, GCN


# --- Utility Function: Accuracy Calculation ---
def accuracy(pred, labels):
    _, predicted = torch.max(pred, dim=1)
    correct = (predicted == labels).sum().item()
    acc = correct / len(labels)
    return acc


def train_and_evaluate(model, data, unique_labels, epochs, lr, patience=50):
    """
    Trains the GCN model, incorporating early stopping.

    Args:
        model (nn.Module): GCN model instance.
        data (Data): PyTorch Geometric graph data object.
        unique_labels (list): Original fault class names.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience value.
    """
    # 🌟 Only keep memory usage print here 🌟
    print_memory_usage("Training and Evaluation - Start")
    print("\n--- 4. Model Training and Evaluation ---")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    # --- Early Stopping Initialization ---
    best_test_acc = 0.0
    patience_counter = 0
    best_model_path = 'best_gcn_checkpoint.pt'  # Best model save path
    # -----------------------

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            train_acc = accuracy(out_eval[data.train_mask], data.y[data.train_mask])
            test_acc = accuracy(out_eval[data.test_mask], data.y[data.test_mask])

        # --- Early Stopping Check ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save the current best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\nEarly stopping triggered! Test accuracy did not improve for {patience} epochs.')
            break
        # -----------------

        if epoch % 1 == 0:
            print(f'Epoch {epoch:02d}/{epochs}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f} '
                  f'(Patience: {patience_counter}/{patience})')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Model Training Time: {elapsed_time:.2f} seconds")

    # Load the best model for final evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loading best model (Test Acc: {best_test_acc:.4f}) for final evaluation.")

    # 🌟 Only keep memory usage print here 🌟
    print_memory_usage("Training and Evaluation - End")

    # Final Evaluation (Classification Report and Confusion Matrix)
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, predicted = torch.max(out, dim=1)

        # Classification Report
        report = classification_report(data.y[data.test_mask].numpy(),
                                       predicted[data.test_mask].numpy(),
                                       target_names=unique_labels,
                                       digits=4)
        print("\nClassification Report:\n", report)

        # Confusion Matrix
        cm = confusion_matrix(data.y[data.test_mask].numpy(),
                              predicted[data.test_mask].numpy())
        print("\nConfusion Matrix:\n", cm)

    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Clean up temporary saved model file
    if os.path.exists(best_model_path):
        os.remove(best_model_path)


# --- Main Execution Flow ---
if __name__ == '__main__':
    # 1. Data Loading and Initial Preprocessing
    nodes, node_labels, all_labels = load_and_segment_data(FILE_PATH, STEP_LENGTH)

    # 2. Call Part 1: PCA Dimensionality Reduction (features.py)
    nodes_pca, n_components, unique_labels = feature_reduction_pca(nodes, node_labels)

    # 3. Call Part 2: Graph Building (graph_model.py)
    data = build_graph_knn(nodes_pca, node_labels, k=KNN_K, pre_update=True)

    # 4. Call Part 3: Graph Neural Network Building (graph_model.py)
    category_count = len(unique_labels)
    model = GCN(in_channels=n_components, out_channels=category_count)

    # 5. Call Part 4: Training and Evaluation with patience=50

    train_and_evaluate(model, data, unique_labels, epochs=EPOCHS, lr=LEARNING_RATE, patience=50)