"""
Simplified GraphBuilder that loads pre-computed features.
Much faster than computing features on-the-fly.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from load_features import load_saved_features, load_graph_structure


class SimpleGraphBuilder:
    """
    Simple graph builder that uses pre-computed features.
    """
    
    def __init__(self, graph_info, features_dict, label, graph_idx):
        """
        Args:
            graph_info: Dict with 'num_nodes', 'edge_list', 'node_labels'
            features_dict: Dict mapping feature names to arrays
            label: Graph label (0 or 1)
            graph_idx: Index of this graph in the dataset
        """
        self.num_nodes = graph_info['num_nodes']
        self.edge_list = graph_info['edge_list']
        self.features_dict = features_dict
        self.label = label
        self.graph_idx = graph_idx
    
    def build(self, selected_features=None):
        """
        Build PyG Data object with selected features.
        
        Args:
            selected_features: List of feature names to use. If None, use all.
        
        Returns:
            Data object
        """
        # Create edge index
        edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
        
        # Select features
        if selected_features is None:
            selected_features = list(self.features_dict.keys())
        
                # Separate matrix features from scalar features
        matrix_features = []
        scalar_features = []
        
        for feat_name in selected_features:
            if feat_name not in self.features_dict:
                raise ValueError(f"Feature {feat_name} not available")
            
            # Get features for this graph
            feat_array = self.features_dict[feat_name][self.graph_idx]
            
            # Only keep features for actual nodes (remove padding)
            feat_array = feat_array[:self.num_nodes]
            
            if len(feat_array.shape) == 2 and feat_array.shape[0] == self.num_nodes:
                # feat_array should be (num_nodes, num_nodes)
                #print(f"DEBUG: adjacency feat_array shape: {feat_array.shape}")
                matrix_features.append(feat_array)
            else:
                # Scalar features
                #print(f"DEBUG: scalar {feat_name} feat_array shape: {feat_array.shape}")
                scalar_features.append(feat_array.reshape(-1, 1))
    
            # Combine features
            if matrix_features and not scalar_features:
                # Only adjacency features
                x = np.concatenate(matrix_features, axis=1)
            elif scalar_features and not matrix_features:
                # Only scalar features
                x = np.hstack(scalar_features)
            elif matrix_features and scalar_features:
                # Both types - concatenate adjacency first, then scalars
                mat_features = np.concatenate(matrix_features, axis=1)
                scal_features = np.hstack(scalar_features)
                x = np.hstack([mat_features, scal_features])
            else:
                # Fallback: use degree
                degrees = np.zeros(self.num_nodes)
                for i, j in self.edge_list:
                    if i < self.num_nodes:
                        degrees[i] += 1
                x = degrees.reshape(-1, 1)
            
                    # Convert to tensor
            x = torch.FloatTensor(x)
            y = torch.tensor(self.label, dtype=torch.long)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes, y=y)
        
        return data


def create_simple_dataset(loop_order, selected_features=None, normalize=True, 
                          data_dir='Graph_Edge_Data', scaler=None, 
                          max_features=None, extra_train=False):
    """
    Create dataset using pre-computed features.
    Args:
        loop_order: Loop order (int) or list of loop orders (e.g., [7,8,9] for training, 10 for testing)
        selected_features: List of feature names to use
        normalize: Whether to normalize features
        scaler: Pre-fitted scaler to use (if None, will fit new scaler when normalize=True)
        max_features: Force a maximum feature dimension (pads/truncates if needed)
        extra_train: If True, load features from 'features_loop_{loop_order}to_extra'
    Returns:
        dataset: List of PyG Data objects
        scaler: StandardScaler object (if normalize=True)
        current_max_features: Maximum feature dimension before padding/truncating
    """
    # Handle single loop order vs multiple loop orders
    loop_orders = [loop_order] if isinstance(loop_order, int) else loop_order

    # Default features
    if selected_features is None:
        selected_features = ['degree', 'betweenness', 'clustering']

    print(f"Loading features: {selected_features}")
    dataset = []
    idx_counter = 0

    for lo in loop_orders:
        # Switch feature source depending on extra_train
        feature_source = f"features_loop_{lo}to" if extra_train else f"features_loop_{lo}"
        print(f"Loading features from {feature_source}...")

        features_dict, labels = load_saved_features(lo, selected_features, data_dir, extra_train=extra_train)
        graph_infos = load_graph_structure(lo, data_dir, extra_train=extra_train)

        for local_idx, (graph_info, label) in enumerate(zip(graph_infos, labels)):
            builder = SimpleGraphBuilder(graph_info, features_dict, label, local_idx)
            data = builder.build(selected_features)
            dataset.append(data)
            idx_counter += 1

    print(f"Created dataset with {len(dataset)} graphs")
    print(f"Feature dimension: {dataset[0].x.shape[1]}")

    # Feature dimension alignment
    feature_dims = [data.x.shape[1] for data in dataset]
    current_max_features = max(feature_dims)
    min_features = min(feature_dims)

    target_features = max_features if max_features is not None else current_max_features

    if target_features != min_features or any(d != target_features for d in feature_dims):
        print(f"Standardizing to {target_features} features")
        for data in dataset:
            if data.x.shape[1] < target_features:
                padding = torch.zeros(data.x.shape[0], target_features - data.x.shape[1])
                data.x = torch.cat([data.x, padding], dim=1)
            elif data.x.shape[1] > target_features:
                data.x = data.x[:, :target_features]

    # Normalize
    if normalize:
        print("Normalizing features...")
        all_features = np.vstack([data.x.numpy() for data in dataset])
        scaler = StandardScaler()
        scaler.fit(all_features)
        for data in dataset:
            data.x = torch.FloatTensor(scaler.transform(data.x.numpy()))

    return dataset, scaler, current_max_features


#Statistics function to quickly summarize dataset
def quick_dataset_stats(dataset):
    """Print quick statistics about the dataset."""
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
