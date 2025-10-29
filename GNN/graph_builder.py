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


def create_simple_dataset(file_ext='7', selected_features=None, normalize=True, 
                          data_dir='Graph_Edge_Data', scaler=None, 
                          max_features=None, n_jobs=None, chunk_size=None):
    """
    Create dataset using pre-computed features.
    Args:
        file_ext: Loop order(s) to load (int or list of int)
        selected_features: List of feature names to use
        normalize: Whether to normalize features
        scaler: Pre-fitted scaler to use (if None, will fit new scaler when normalize=True)
        max_features: Force a maximum feature dimension (pads/truncates if needed)
    Returns:
        dataset: List of PyG Data objects
        scaler: StandardScaler object (if normalize=True)
        current_max_features: Maximum feature dimension before padding/truncating
    """

    # Default features
    if selected_features is None:
        selected_features = ['degree', 'betweenness', 'clustering']

    print(f"Loading features: {selected_features}")
    dataset = []
    idx_counter = 0
    feature_source =  f"features_loop_{file_ext}"
    print(f"Loading features from {feature_source}...")

    features_dict, labels = load_saved_features(file_ext, 
                                                selected_features, 
                                                data_dir, n_jobs=n_jobs,
                                                chunk_size=chunk_size
                                                )
    graph_infos = load_graph_structure(file_ext, data_dir, n_jobs=n_jobs, chunk_size=chunk_size)

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

    if max_features is None or current_max_features > max_features:
        max_features = current_max_features
    target_features = max_features


    if any(d != target_features for d in feature_dims):
        print(f"Standardizing to {target_features} features (padding only)")
        for data in dataset:
            if data.x.shape[1] < target_features:
                padding = torch.zeros(data.x.shape[0], target_features - data.x.shape[1])
                data.x = torch.cat([data.x, padding], dim=1)

    # Normalize
    if normalize:
        print("Normalizing features...")
        all_features = np.vstack([data.x.numpy() for data in dataset])
        scaler = StandardScaler()
        scaler.fit(all_features)
        for data in dataset:
            data.x = torch.FloatTensor(scaler.transform(data.x.numpy()))

    return dataset, scaler, current_max_features

import itertools
import numpy as np

def print_dataset_stats(ds, name="Dataset"):
    """Print quick statistics about a dataset (handles ConcatDataset)."""
    if isinstance(ds, torch.utils.data.ConcatDataset):
        graphs = list(itertools.chain.from_iterable(ds.datasets))
    else:
        graphs = list(ds)

    num_graphs = len(graphs)
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.shape[1] // 2 for g in graphs]
    labels = [g.y.item() for g in graphs]
    feat_dims = sorted({g.x.shape[1] for g in graphs})

    print(f"\n{name} statistics:")
    print(f"  Number of graphs: {num_graphs}")
    print(f"  Nodes: mean {np.mean(num_nodes):.1f} (min {min(num_nodes)}, max {max(num_nodes)})")
    print(f"  Edges: mean {np.mean(num_edges):.1f}")
    print(f"  Labels: {sum(labels)} positive, {num_graphs - sum(labels)} negative")
    print(f"  Feature dimensions: {feat_dims}")


if __name__ == "__main__":
    # Example usage
    print("Testing simple dataset creation...")
    
    # Create dataset with specific features
    dataset, scaler = create_simple_dataset(
        file_ext='7',
        selected_features=['degree', 'betweenness', 'clustering'],
        normalize=True
    )
    
    
