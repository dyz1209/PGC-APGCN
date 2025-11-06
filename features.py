# features.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from config import print_memory_usage  # Import utility tool


def load_and_segment_data(file_path, step_length):
    """
    Loads data and segments it by step_length to create node features and labels.
    """
    data_df = pd.read_csv(file_path)
    labels = data_df.columns.values  # Column names are the fault labels

    nodes = []
    node_labels = []
    # Iterate through each column (each fault type)
    for col in data_df.columns:
        feature_data = data_df[col].values
        num_nodes = len(feature_data) // step_length
        for i in range(num_nodes):
            start_idx = i * step_length
            segment = feature_data[start_idx:start_idx + step_length]
            nodes.append(segment)
            node_labels.append(col)  # Use column name as the segment's label

    nodes = np.array(nodes)
    print(f"Data loading complete: Total nodes={nodes.shape[0]}, Original dimension={nodes.shape[1]}")
    return nodes, node_labels, labels


def feature_reduction_pca(nodes, initial_labels):
    """
    Standardizes data and performs PCA for dimensionality reduction,
    automatically selecting components based on explained variance ratio threshold.

    Returns:
        tuple: (nodes_pca, n_components, unique_labels)
    """
    # print_memory_usage("PCA Dimensionality Reduction - Start")
    print("\n--- 1. Feature Reduction (PCA) ---")

    # Standardize data
    scaler = StandardScaler()
    nodes_scaled = scaler.fit_transform(nodes)

    # First PCA: Calculate explained variance ratio
    pca_full = PCA()
    pca_full.fit(nodes_scaled)
    explained_variance_ratio = pca_full.explained_variance_ratio_

    # Calculate threshold and select number of components
    total_components = len(explained_variance_ratio)
    threshold = 1 / total_components
    n_components = sum(explained_variance_ratio > threshold)
    if n_components == 0:
        n_components = 1  # Ensure at least one component is kept

    print(f"Selected to keep {n_components} principal components.")

    # Second PCA: Perform dimensionality reduction
    pca = PCA(n_components=n_components)
    nodes_pca = pca.fit_transform(nodes_scaled)

    print("PCA reduced node feature shape:", nodes_pca.shape)
    # print_memory_usage("PCA Dimensionality Reduction - End")

    # Plot explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(explained_variance_ratio, marker='o', linestyle='--')
    # plt.axhline(y=threshold, color='r', linestyle='-', label=f'Threshold ({threshold:.4f})') <-- Removed this line
    plt.title('Principal Component Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance')
    # plt.legend() <-- Removed this line to fix the UserWarning
    plt.grid(True)
    # plt.savefig('pca_variance_ratio.pdf', format='pdf')
    plt.show()

    unique_labels = np.unique(initial_labels)
    return nodes_pca, n_components, unique_labels
