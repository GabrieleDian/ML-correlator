# GraphBuilder_with_features.py

## Overview
This module handles the conversion of planar graphs into feature-rich PyTorch Geometric Data objects. It extracts specialized features designed for planar graph classification.

## Features

### Feature Groups

1. **Basic Features** (`basic`)
   - Node degree

2. **Face Features** (`face`)
   - Number of faces containing each node
   - Average face size
   - Maximum face size
   - Face size variance

3. **Spectral Features** 
   - **Node-level** (`spectral_node`): Fiedler vector, eigenvector energy, third eigenvector
   - **Graph-level** (`spectral_global`): Algebraic connectivity, spectral gap, largest eigenvalue

4. **Dual Graph Features** (`dual`)
   - Dual degree and clustering
   - Dual degree ratio
   - Dual betweenness
   - Face-edge ratio

5. **Centrality Features** (`centrality`)
   - Betweenness centrality
   - Closeness centrality
   - Eigenvector centrality
   - Clustering coefficient
   - PageRank

6. **Laplacian Positional Encoding** (`laplacian_pe`)
   - Configurable number of eigenvectors (k)

## Usage

### Basic Usage
```python
from GraphBuilder_with_features import GraphBuilder

# Define graph edges
edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ...]

# Create builder with default features
builder = GraphBuilder(
    solid_edges=edges,
    coeff=1,  # Target label
    selected_features=['basic', 'face', 'spectral_node', 'centrality'],
    laplacian_pe_k=3
)

# Build graph
data = builder.build()
```

### Dataset Creation
```python
from GraphBuilder_with_features import create_graph_dataset

# Prepare graph data
graphs_data = [(edges1, label1), (edges2, label2), ...]

# Configure features
feature_config = {
    'selected_features': ['basic', 'face', 'spectral_node', 'centrality'],
    'laplacian_pe_k': 3,
    'include_global_features': False
}

# Create dataset with normalization
dataset, scaler = create_graph_dataset(graphs_data, feature_config)
```

## Feature Selection

### Recommended Configurations

**Minimal (5 features)**:
```python
selected_features = ['basic', 'face']
```

**Balanced (13 features)**:
```python
selected_features = ['basic', 'face', 'spectral_node', 'centrality']
```

**Full (18+ features)**:
```python
selected_features = ['basic', 'face', 'spectral_node', 'dual', 'centrality']
```

## Feature Dimensions

| Feature Group | Dimensions | Description |
|--------------|------------|-------------|
| basic | 1 | Node degree |
| face | 4 | Planar face properties |
| spectral_node | 3 | Node-level spectral features |
| spectral_global | 3 | Graph-level spectral (if included) |
| dual | 5 | Dual graph properties |
| centrality | 5 | Various centrality measures |
| laplacian_pe | k | k eigenvectors |

## Important Notes

1. **Planar Graphs Only**: Face and dual features require planar graphs
2. **Normalization**: `create_graph_dataset` automatically normalizes features
3. **Feature Names**: Access via `data.feature_names`
4. **Global Features**: Can be included as node features with `include_global_features=True`

## Example: Custom Feature Selection
```python
# Get feature information
builder = GraphBuilder(edges, label)
info = builder.get_feature_info()
print(f"Total dimensions: {info['total_dimensions']}")
print(f"Selected features: {info['selected_features']}")

# Access feature names after building
data = builder.build()
print(f"Feature names: {data.feature_names}")
```

## Tips
- Start with 'balanced' configuration
- Face features are crucial for planar graphs
- Spectral features help with graph structure
- Too many features can cause overfitting on small datasets