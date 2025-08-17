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


class GINNet(nn.Module):
    """
    Graph Isomorphism Network - More powerful than GCN for graph-level tasks
    GIN is provably as powerful as the WL test for graph isomorphism
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.2, num_layers=3):
        super(GINNet, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Jumping knowledge - combine features from all layers
        self.jump = nn.Linear(num_layers * hidden_dim, hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        # Store representations from each layer
        layer_representations = []
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Store layer representation
            layer_representations.append(x)
        
        # Jumping knowledge connection
        x = torch.cat(layer_representations, dim=-1)
        x = self.jump(x)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_add_pool(x, batch)
        return self.classifier(x)


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


class HybridGNN(nn.Module):
    """
    Hybrid architecture combining multiple convolution types
    Captures different aspects of graph structure
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.2):
        super(HybridGNN, self).__init__()
        
        self.dropout = dropout
        
        # Different types of graph convolutions
        self.gcn = GCNConv(num_features, hidden_dim)
        self.gin = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.gat = GATConv(num_features, hidden_dim // 2, heads=2)
        
        # Combine different representations
        self.combine = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Second layer
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        
        # Pooling layers
        self.pool1 = SAGPooling(hidden_dim, ratio=0.8)
        
        # Final layers
        self.global_pool = nn.Linear(hidden_dim * 3, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply different convolutions
        x_gcn = F.relu(self.gcn(x, edge_index))
        x_gin = F.relu(self.gin(x, edge_index))
        x_gat = F.relu(self.gat(x, edge_index))
        
        # Concatenate representations
        x = torch.cat([x_gcn, x_gin, x_gat], dim=-1)
        x = self.combine(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Hierarchical pooling (optional)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
        # Multiple global pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Combine pooling results
        x = torch.cat([x_mean, x_max, x_add], dim=-1)
        x = self.global_pool(x)
        
        return self.classifier(x)


class PlanarGNN(nn.Module):
    """
    Specialized architecture for planar graphs
    Incorporates domain-specific inductive biases
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.2):
        super(PlanarGNN, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Planar-aware convolutions (2 layers optimal for small graphs)
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        )
        
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        )
        
        # Edge features processing (for planar graphs)
        self.edge_attr_lin = nn.Linear(1, hidden_dim)
        
        # Skip connections
        self.skip_lin = nn.Linear(num_features + hidden_dim * 2, hidden_dim)
        
        # Readout with attention
        self.att_weight = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.att_weight)
        
        # Final classifier with residual
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Store input for skip connection
        x_input = x
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # First convolution
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second convolution
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        
        # Skip connection combining input and both conv outputs
        x_skip = torch.cat([x_input, x1, x2], dim=-1)
        x_skip = self.skip_lin(x_skip)
        
        # Attention-based readout
        att_scores = torch.matmul(x_skip, self.att_weight.t())
        att_scores = softmax(att_scores, batch, dim=0)
        
        # Weighted and unweighted pooling
        x_att = global_add_pool(x_skip * att_scores, batch)
        x_mean = global_mean_pool(x_skip, batch)
        
        # Combine both pooling strategies
        x = torch.cat([x_att, x_mean], dim=-1)
        
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
        'gin': GINNet,
        'gat': GATNet,
        'hybrid': HybridGNN,
        'planar': PlanarGNN,
        'simple': SimpleButEffectiveGNN
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
        'GIN (Recommended)': create_gnn_model('gin', num_features, hidden_dim=64, num_layers=3),
        'GAT': create_gnn_model('gat', num_features, hidden_dim=64, num_heads=4),
        'Hybrid': create_gnn_model('hybrid', num_features, hidden_dim=48),
        'Planar-Specific': create_gnn_model('planar', num_features, hidden_dim=64),
        'Simple but Effective': create_gnn_model('simple', num_features, hidden_dim=32)
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