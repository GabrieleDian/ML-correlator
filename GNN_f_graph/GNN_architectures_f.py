"""
Advanced GNN architectures for planar graph classification
Designed for graphs with 11-14 nodes to break through the 70-80% accuracy barrier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (RGCNConv,HeteroConv, GCNConv,global_mean_pool)

# RGCN model for multi-edge graphs
class GraphBinaryClassifier(torch.nn.Module):
    def __init__(self, num_features, num_relations, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = RGCNConv(num_features, hidden_dim, num_relations)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)  # logits

    def forward(self, x, edge_index, edge_type, batch=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_type)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_type)))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)

class HeteroGraphBinaryClassifier(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, dropout=0.2):
        super().__init__()
        # learnable initial node embeddings
        self.node_emb = torch.nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = HeteroConv({
            'type1': GCNConv(hidden_dim, hidden_dim),
            'type2': GCNConv(hidden_dim, hidden_dim)
        }, aggr='sum')
        self.conv2 = HeteroConv({
            'type1': GCNConv(hidden_dim, hidden_dim),
            'type2': GCNConv(hidden_dim, hidden_dim)
        }, aggr='sum')
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data, batch= None):
        x = self.node_emb(data['node'].x)  # or just integer indices
        x_dict = {'node': x}
        x_dict = self.conv(x_dict, data.edge_index_dict)
        x = x_dict['node']
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))
def create_gnn_model(architecture='rgcn', num_features=9, hidden_dim=64, dropout=0.2,num_nodes= None, **kwargs):
    """
    Factory function to create different GNN architectures, including multi-edge graphs
    """
    # Architecture mapping
    architectures = {
        'rgcn'  : GraphBinaryClassifier,
        'hetero': HeteroGraphBinaryClassifier}        # RGCN for multi-edg}

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model_class = architectures[architecture]

    # Filter kwargs for each architecture
    if architecture == 'rgcn':
        valid_params = ['num_relations','num_node_features']  # Number of edge types
    elif architecture == 'hetero':
        valid_params = ['num_nodes']      # Needed for learnable node embeddings
    else:
        valid_params = []

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_class(
        hidden_dim=hidden_dim, # <-- important for multi-edge graphs
        dropout=dropout,
        num_relations=2,  # <-- important for multi-edge graphs
        num_features = num_features,
        **filtered_kwargs
    )



# Example usage and comparison
if __name__ == "__main__":
    # Example parameters
    num_features = 26  # From your feature extractor
    batch_size = 32
    
    # Create different models
    models = {
        'R-GCN': create_gnn_model('rgcn', num_features, hidden_dim=64, num_relations=2, num_layers=2),
        'Hetero-GCN': create_gnn_model('hetero', num_features, hidden_dim=64, num_nodes=100, num_layers=2)
    }
    
    # Print model summaries
    print("Model Architectures Summary:")
    print("="*60)
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {num_params:,} parameters")
 