"""
Simplified GraphBuilder that loads pre-computed features.
Much faster than computing features on-the-fly.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from load_features_f import load_saved_features, load_graph_structure


import torch
from torch_geometric.data import Data
import numpy as np

class SimpleGraphBuilder:
    """
    Graph builder that combines scalar and matrix node features for multi-edge graphs,
    and includes edge type information.
    """
    
    def __init__(self, graph_info, features_dict, label, graph_idx):
        """
        Args:
            graph_info: Dict with 'num_nodes', 'edge_list', 'edge_types', 'node_labels'
            features_dict: Dict mapping feature names to numpy arrays
            label: Graph label (0 or 1)
            graph_idx: Index of this graph in the dataset
        """
        self.num_nodes = graph_info['num_nodes']
        self.edge_list = graph_info['edge_list']
        self.edge_types = graph_info['edge_types']  # 0 = denominator, 1 = numerator
        self.features_dict = features_dict
        self.label = label
        self.graph_idx = graph_idx
    
    def build(self, selected_features=None):
        """
        Build PyG Data object with selected features.
        
        Args:
            selected_features: List of feature names to include. Defaults to all.
        
        Returns:
            Data object (PyTorch Geometric)
        """
        # Edge index tensor
        edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
        # Edge type tensor as edge_attr
        edge_attr = torch.tensor(self.edge_types, dtype=torch.float).unsqueeze(1)  # shape [num_edges,1]
        
        # Node features
        if selected_features is None:
            selected_features = list(self.features_dict.keys())
        
        matrix_features = []
        scalar_features = []
        
        for feat_name in selected_features:
            if feat_name not in self.features_dict:
                raise ValueError(f"Feature {feat_name} not available")
            
            feat_array = self.features_dict[feat_name][self.graph_idx]
            feat_array = feat_array[:self.num_nodes]  # remove padding
            
            if len(feat_array.shape) == 2 and feat_array.shape[0] == self.num_nodes:
                # Matrix feature
                matrix_features.append(feat_array)
            else:
                # Scalar feature
                scalar_features.append(feat_array.reshape(-1, 1))
        
        # Combine node features
        if matrix_features and not scalar_features:
            x = np.concatenate(matrix_features, axis=1)
        elif scalar_features and not matrix_features:
            x = np.hstack(scalar_features)
        elif matrix_features and scalar_features:
            x = np.hstack([np.concatenate(matrix_features, axis=1),
                           np.hstack(scalar_features)])
        else:
            # Fallback: degree feature
            degrees = np.zeros(self.num_nodes)
            for i, j in self.edge_list:
                if i < self.num_nodes:
                    degrees[i] += 1
            x = degrees.reshape(-1, 1)
        
        x = torch.FloatTensor(x)
        y = torch.tensor(self.label, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.num_nodes, y=y)
        return data



from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def create_simple_dataset(loop_order, selected_features=None, normalize=True, data_dir='Graph_Edge_Data', scaler=None, max_features=None):
    """
    Create dataset using pre-computed features and multi-edge graphs.
    
    Args:
        loop_order: int or list of loop orders
        selected_features: List of feature names to use
        normalize: Whether to normalize node features
        scaler: Pre-fitted StandardScaler to use (if None, will fit new scaler)
        max_features: Force all graphs to have this number of features (pad/truncate)
    
    Returns:
        dataset: List of PyG Data objects
        scaler: StandardScaler object (if normalize=True)
        feature_dim: Number of features per node
    """
    if isinstance(loop_order, int):
        loop_orders = [loop_order]
    else:
        loop_orders = loop_order

    if selected_features is None:
        selected_features = ['degree', 'betweenness', 'clustering']

    print(f"Loading features: {selected_features}")
    dataset = []

    for lo in loop_orders:
        # Load node-level features and labels
        features_dict, labels = load_saved_features(lo, selected_features, data_dir)
        # Load graph structure (includes edge_types)
        graph_infos = load_graph_structure(lo, data_dir)
        print(f"Loaded {len(graph_infos)} graphs for loop order {lo}")

        for idx, (graph_info, label) in enumerate(zip(graph_infos, labels)):
            builder = SimpleGraphBuilder(graph_info, features_dict, label, idx)
            data = builder.build(selected_features)
            dataset.append(data)

    print(f"Created dataset with {len(dataset)} graphs")

    # Determine target feature dimension
    feature_dims = [data.x.shape[1] for data in dataset]
    current_max_features = max(feature_dims)
    target_features = max_features if max_features is not None else current_max_features

    if any(d != target_features for d in feature_dims):
        print(f"Padding/truncating features to {target_features}")
        for data in dataset:
            if data.x.shape[1] < target_features:
                padding = torch.zeros(data.x.shape[0], target_features - data.x.shape[1])
                data.x = torch.cat([data.x, padding], dim=1)
            elif data.x.shape[1] > target_features:
                data.x = data.x[:, :target_features]

    # Normalize node features if requested (edge_attr is untouched)
        if normalize:
            print("Normalizing node features...")

            all_features = np.vstack([data.x.numpy() for data in dataset])
            if scaler is None:
                scaler = StandardScaler()
            scaler.fit(all_features)

            # Transform
            for data in dataset:
                data.x = torch.FloatTensor(scaler.transform(data.x.numpy()))


    return dataset, scaler, target_features


def quick_dataset_stats(dataset):
    """Print quick statistics about the dataset, including edge types."""
    num_graphs = len(dataset)
    num_nodes = [data.num_nodes for data in dataset]
    num_edges = [data.edge_index.shape[1] // 2 for data in dataset]
    labels = [data.y.item() for data in dataset]
    
    print(f"\nDataset Statistics:")
    print(f"  Number of graphs: {num_graphs}")
    print(f"  Average nodes: {np.mean(num_nodes):.1f} (min: {min(num_nodes)}, max: {max(num_nodes)})")
    print(f"  Average edges: {np.mean(num_edges):.1f}")
    print(f"  Label distribution: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    print(f"  Feature dimension: {dataset[0].x.shape[1]}")

    # Optional: summarize edge types if available
    if hasattr(dataset[0], 'edge_attr') and dataset[0].edge_attr is not None:
        den_edges = []
        num_edges_count = []
        for data in dataset:
            den_edges.append((data.edge_attr == 0).sum().item() // 2)
            num_edges_count.append((data.edge_attr == 1).sum().item() // 2)
        print(f"  Average denominator edges: {np.mean(den_edges):.1f}")
        print(f"  Average numerator edges: {np.mean(num_edges_count):.1f}")


if __name__ == "__main__":
    # Example usage
    print("Testing simple dataset creation...")
    
    # Create dataset with specific features
    dataset, scaler = create_simple_dataset(
        loop_order=8,
        normalize=True
    )
    
    # Print statistics
    quick_dataset_stats(dataset)
    
    # Show first graph
    print(f"\nFirst graph:")
    print(f"  Nodes: {dataset[0].num_nodes}")
    print(f"  Edges: {dataset[0].edge_index.shape}")
    print(f"  Features shape: {dataset[0].x.shape}")
    print(f"  Label: {dataset[0].y.item()}")
