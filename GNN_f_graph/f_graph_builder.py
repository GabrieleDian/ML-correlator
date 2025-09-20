"""
Simplified GraphBuilder that loads pre-computed features.
Much faster than computing features on-the-fly.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from load_features_f import load_saved_features, load_graph_structure


import torch
from torch_geometric.data import Data
import numpy as np

class SimpleGraphBuilder:
    def __init__(self, graph_info, features_dict, label, graph_idx):
        self.num_nodes = graph_info['num_nodes']
        self.edge_list = graph_info['edge_list']
        self.edge_types = graph_info['edge_types']  # 0 or 1
        self.features_dict = features_dict
        self.label = label
        self.graph_idx = graph_idx

    def build(self, selected_features=None):
        # Node features
        if selected_features is None:
            selected_features = list(self.features_dict.keys())

        matrix_features = []
        scalar_features = []

        for feat_name in selected_features:
            feat_array = self.features_dict[feat_name][self.graph_idx][:self.num_nodes]
            if len(feat_array.shape) == 2 and feat_array.shape[0] == self.num_nodes:
                matrix_features.append(feat_array)
            else:
                scalar_features.append(feat_array.reshape(-1,1))

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
            for i,j in self.edge_list:
                if i < self.num_nodes:
                    degrees[i] += 1
            x = degrees.reshape(-1,1)

        x = torch.FloatTensor(x)
        y = torch.tensor([self.label], dtype=torch.float)  # BCE compatible

        # Regular Data for R-GCN
        edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(self.edge_types, dtype=torch.long)  # must be long
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                    y=y, num_nodes=self.num_nodes)

        return data


def create_simple_dataset_rgcn(file_ext='7', selected_features=None, normalize=True,
                               data_dir='../Graph_Edge_Data', scaler=None,
                               max_features=None, batch_size=32):
    """
    Create dataset ready for R-GCN (batched graphs with edge types)
    """
    if selected_features is None:
        selected_features = ['degree', 'betweenness', 'clustering']

    print(f"Loading features: {selected_features}")
    dataset = []
    
    features_dict, labels = load_saved_features(file_ext, selected_features, data_dir)
    graph_infos = load_graph_structure(file_ext, data_dir)

    for idx, (graph_info, label) in enumerate(zip(graph_infos, labels)):
        builder = SimpleGraphBuilder(graph_info, features_dict, label, idx)
        data = builder.build()  # R-GCN
        dataset.append(data)

    # Feature dimension alignment
    feature_dims = [data.x.shape[1] for data in dataset]
    target_features = max_features if max_features is not None else max(feature_dims)

    for data in dataset:
        if data.x.shape[1] < target_features:
            padding = torch.zeros(data.x.shape[0], target_features - data.x.shape[1])
            data.x = torch.cat([data.x, padding], dim=1)
        elif data.x.shape[1] > target_features:
            data.x = data.x[:, :target_features]

    # Normalize features
    if normalize:
        print("Normalizing features...")
        all_features = np.vstack([data.x.numpy() for data in dataset])
        scaler = StandardScaler()
        scaler.fit(all_features)
        for data in dataset:
            data.x = torch.FloatTensor(scaler.transform(data.x.numpy()))
    
    return dataset, scaler, target_features

#Statistics function to quickly summarize dataset
import numpy as np

def quick_dataset_stats(dataset):
    """Print quick statistics for dataset or Subset."""
    
    # If dataset is a Subset, get the underlying dataset
    if isinstance(dataset, torch.utils.data.dataset.Subset):
        dataset = dataset.dataset

    num_graphs = len(dataset)
    num_nodes = [data.num_nodes for data in dataset]
    num_edges = [data.edge_index.shape[1] for data in dataset]
    labels = [data.y.item() for data in dataset]

    print(f"\nDataset Statistics:")
    print(f"  Number of graphs: {num_graphs}")
    print(f"  Average nodes: {np.mean(num_nodes):.1f} (min: {min(num_nodes)}, max: {max(num_nodes)})")
    print(f"  Average edges: {np.mean(num_edges):.1f}")
    print(f"  Label distribution: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    print(f"  Feature dimension: {dataset[0].x.shape[1]}")

    # Summarize edge types
    if hasattr(dataset[0], 'edge_type') and dataset[0].edge_type is not None:
        type0_counts = [(data.edge_type == 0).sum().item() for data in dataset]
        type1_counts = [(data.edge_type == 1).sum().item() for data in dataset]
        print(f"  Average type 0 edges: {np.mean(type0_counts):.1f}")
        print(f"  Average type 1 edges: {np.mean(type1_counts):.1f}")

if __name__ == "__main__":
    # Example usage
    print("Testing simple dataset creation...")
    
    # Create dataset with specific features
    dataset, scaler = create_simple_dataset_rgcn(
        file_ext='7',
        selected_features=['degree', 'betweenness', 'clustering'],
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

