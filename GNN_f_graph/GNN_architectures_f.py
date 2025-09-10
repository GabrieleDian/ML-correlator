"""
Advanced GNN architectures for planar graph classification
Designed for graphs with 11-14 nodes to break through the 70-80% accuracy barrier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv, GATConv, GCNConv, GraphConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    SAGPooling
)
from torch_geometric.utils import  softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_add_pool

class GINNetMultiEdge(nn.Module):
    """
    Edge-aware GIN-like network for graph regression (predict graph-level real label).
    """
    def __init__(self, num_features, edge_attr_dim=1, hidden_dim=64, num_classes=1, dropout=0.2, num_layers=3):
        super(GINNetMultiEdge, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = num_classes
        # Build NNConv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            nn_edge = nn.Sequential(
                nn.Linear(edge_attr_dim, hidden_dim * in_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim * in_dim, hidden_dim * in_dim)
            )
            conv = NNConv(in_dim, hidden_dim, nn_edge, aggr='add')
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Jumping knowledge
        self.jump = nn.Linear(num_layers * hidden_dim, hidden_dim)
        
        # Final regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # single scalar output
        )
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        layer_reps = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_reps.append(x)
        
        # Jumping knowledge
        x = torch.cat(layer_reps, dim=-1)
        x = self.jump(x)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_add_pool(x, batch)
        return self.regressor(x)  # regression output



class GATNet(nn.Module):
    """
    Graph Attention Network - Uses attention mechanism to weight neighbor importance
    Better for graphs where different edges have different importance
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, 
                 dropout=0.2, num_heads=4, num_layers=2):
        super(GATNet, self).__init__()
        
        self.dropout = dropout
        
        # Multi-head attention layers
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        if num_layers > 2:
            self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        else:
            self.conv3 = None
        
        # Global attention pooling
        self.global_att = nn.Linear(hidden_dim, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        # First GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Second GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Optional third layer
        if self.conv3 is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            x = F.elu(x)
        
        # Global attention pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Compute attention scores
        att = self.global_att(x)
        att = softmax(att, batch, dim=0)
        
        # Weighted pooling
        x = x * att
        x = global_add_pool(x, batch)
        
        return self.classifier(x)


class SimpleButEffectiveGNN(nn.Module):
    """
    Simple architecture that often works best for small graphs
    Based on the principle that simpler is better for limited data
    """
    def __init__(self, num_features, hidden_dim=32, num_classes=2, dropout=0.1):
        super(SimpleButEffectiveGNN, self).__init__()
        
        # Just two GIN layers with batch norm
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Two layers only (optimal for small graphs)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # Combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        return self.classifier(x)


# Utility function to select architecture
def create_gnn_model(architecture='gin', num_features=26, hidden_dim=64, 
                     num_classes=2, dropout=0.2, **kwargs):
    """
    Factory function to create different GNN architectures
    
    Args:
        architecture: One of ['gin', 'gat', 'hybrid', 'planar', 'simple']
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        dropout: Dropout rate
        **kwargs: Additional architecture-specific parameters
    """
    
    architectures = {
        'gat': GATNet,
        'simple': SimpleButEffectiveGNN,
        'f-graph' : GINNetMultiEdge
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model_class = architectures[architecture]
    
    # Filter kwargs for each architecture
    if architecture == 'gin':
        valid_params = ['num_layers']
    elif architecture == 'gat':
        valid_params = ['num_heads', 'num_layers']
    else:
        valid_params = []
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return model_class(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        **filtered_kwargs
    )


# Example usage and comparison
if __name__ == "__main__":
    # Example parameters
    num_features = 26  # From your feature extractor
    batch_size = 32
    
    # Create different models
    models = {
        'GAT': create_gnn_model('gat', num_features, hidden_dim=64, num_heads=4),
        'Simple but Effective': create_gnn_model('simple', num_features, hidden_dim=32),
        'Edge-aware GIN': create_gnn_model('f-graph', num_features, hidden_dim=64, edge_attr_dim=1, num_layers=3)
    }
    
    # Print model summaries
    print("Model Architectures Summary:")
    print("="*60)
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {num_params:,} parameters")
    
    print("\nRecommendations for your use case:")
    print("1. Start with GIN - it's theoretically more powerful than GCN")
    print("2. Use 2-3 layers only (more causes oversmoothing)")
    print("3. Try the 'Simple but Effective' model - often best for small graphs")
    print("4. Experiment with different pooling strategies")
    print("5. Consider ensemble of different architectures")