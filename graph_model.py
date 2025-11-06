# graph_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
import numpy as np
from config import print_memory_usage


# --- Utility Function: Node Feature Pre-Update ---
def _update_node_features_knn(data, k):
    """
    [Restored Original Logic] Updates node features using the mean of first and second-layer neighbors.
    This was the key step to improve performance in the original code.
    """
    x, edge_index = data.x, data.edge_index
    num_nodes = x.shape[0]
    updated_features = x.clone()

    for node_idx in range(num_nodes):
        # Step 1: Find first-layer neighbors (K-NN)
        first_neighbors_mask = (edge_index[0] == node_idx)
        first_neighbors = edge_index[1][first_neighbors_mask].unique()
        # Exclude self and take the first k neighbors
        first_neighbors = first_neighbors[first_neighbors != node_idx][:k]

        # Step 2: Find second-layer neighbors
        second_neighbors = set()
        for neighbor in first_neighbors.tolist():
            neighbor_neighbors_mask = (edge_index[0] == neighbor)
            neighbor_neighbors = edge_index[1][neighbor_neighbors_mask].unique()
            second_neighbors.update(neighbor_neighbors.tolist())

        # Remove first-layer neighbors and self to get true second-layer neighbors
        second_neighbors = list(second_neighbors - set(first_neighbors.tolist()) - {node_idx})
        second_neighbors_tensor = torch.tensor(second_neighbors[:k], dtype=torch.long)

        # Step 3: Calculate weighted mean
        first_layer_features = x[first_neighbors]
        second_layer_features = x[second_neighbors_tensor]

        # Ensure feature matrices are not empty
        f1_mean = first_layer_features.mean(dim=0) if len(first_layer_features) > 0 else torch.zeros_like(x[0])
        f2_mean = second_layer_features.mean(dim=0) if len(second_layer_features) > 0 else torch.zeros_like(x[0])

        # Update node features: combined mean
        if len(first_layer_features) > 0 or len(second_layer_features) > 0:
            combined_features = (f1_mean + f2_mean) / 2
            updated_features[node_idx] = combined_features
        # Otherwise, keep it unchanged (though theoretically no isolated nodes due to K-NN graph building)

    data.x = updated_features
    return data


# --- Utility Function: Dynamic Graph Modification ---
def delete_min_weighted_edges(node, edge_index, edge_weight):
    """Deletes the edge with the minimum weight connected to the given node (used in GCN forward)."""
    # ⚠️ Note: Dynamic graph modification during GCN training should typically be avoided.
    # This is kept only to maintain the logic of the original code.
    connected_edges_mask = (edge_index[0] == node)
    connected_edge_indices = torch.nonzero(connected_edges_mask).squeeze()

    if connected_edge_indices.dim() == 0:
        return edge_index, edge_weight

    connected_edge_indices = connected_edge_indices.view(-1)
    connected_edge_weights = edge_weight[connected_edge_indices]

    min_weight_idx_local = torch.argmin(connected_edge_weights).item()
    min_weight_idx_global = connected_edge_indices[min_weight_idx_local].item()

    # Delete the edge
    edge_index = torch.cat([edge_index[:, :min_weight_idx_global],
                            edge_index[:, min_weight_idx_global + 1:]], dim=1)
    edge_weight = torch.cat([edge_weight[:min_weight_idx_global],
                             edge_weight[min_weight_idx_global + 1:]], dim=0)

    return edge_index, edge_weight


def build_graph_knn(nodes_pca, node_labels, k=5, pre_update=True):
    """
    Builds the K-NN graph categorized by labels based on PCA features and creates a PyTorch Geometric Data object.
    """
    # print_memory_usage("Graph Building - Start")
    print("\n--- 2. Graph Structure Building (PGC) ---")

    categories = np.unique(node_labels)
    num_nodes = nodes_pca.shape[0]
    edges = []
    edge_weights = []

    # Build K-NN edges within each category
    for category in categories:
        category_indices = [i for i, label in enumerate(node_labels) if label == category]
        category_nodes_pca = nodes_pca[category_indices]

        if len(category_nodes_pca) < k + 1: continue

        tree = BallTree(category_nodes_pca)
        distances, indices = tree.query(category_nodes_pca, k=k + 1)

        for i in range(len(category_nodes_pca)):
            current_global_idx = category_indices[i]
            for j in range(1, k + 1):
                neighbor_idx_local = indices[i][j]
                neighbor_global_idx = category_indices[neighbor_idx_local]
                distance = distances[i][j]

                # Create undirected edge
                edges.append([current_global_idx, neighbor_global_idx])
                edges.append([neighbor_global_idx, current_global_idx])
                # Weight: inverse of distance
                weight = 1 / (distance + 1e-6)
                edge_weights.append(weight)
                edge_weights.append(weight)

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(nodes_pca, dtype=torch.float)

    # Node labels
    unique_labels = np.unique(node_labels)
    label_dict = {label: idx for idx, label in enumerate(unique_labels)}
    node_labels_idx = [label_dict[label] for label in node_labels]
    y = torch.tensor(node_labels_idx, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # Split into training and testing sets
    train_indices, test_indices = train_test_split(
        np.arange(num_nodes), test_size=0.2, random_state=42, stratify=node_labels_idx
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    # Feature pre-update
    if pre_update:
        data = _update_node_features_knn(data, k=k)
        print("Node features updated using K-NN neighbor mean.")

    # print_memory_usage("Graph Building - End")
    return data


# --- GCN Model Definition ---
class GCN(nn.Module):
    """Two-layer GCN model definition."""

    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        print("\n--- 3. Graph Neural Network Model Building (AP-GCN) ---")
        # First Graph Convolution layer
        self.conv1 = GCNConv(in_channels, 64)
        # Second Graph Convolution layer
        self.conv2 = GCNConv(64, out_channels)
        print(f"GCN Model Initialized: Input Dim={in_channels}, Output Classes={out_channels}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr

        # First Graph Convolution layer
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)

        # Dynamic Graph Specification: executed only for the first node (retaining original code logic)
        data.edge_index, data.edge_attr = delete_min_weighted_edges(0, edge_index, edge_weight)

        # Second Graph Convolution layer
        x = self.conv2(x, data.edge_index, data.edge_attr)
        return x