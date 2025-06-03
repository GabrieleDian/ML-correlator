# GNN_architectures.py

## Overview
This module provides various Graph Neural Network architectures optimized for planar graph classification tasks, particularly for graphs with 11-14 nodes.

## Available Architectures

### 1. GINNet (Graph Isomorphism Network)
- **Best for**: General graph-level tasks
- **Key features**: Theoretically as powerful as WL-test, jumping knowledge connections
- **Parameters**: `num_layers` (default: 3)

### 2. GATNet (Graph Attention Network)
- **Best for**: Graphs where edge importance varies
- **Key features**: Multi-head attention, global attention pooling
- **Parameters**: `num_heads` (default: 4), `num_layers` (default: 2)

### 3. HybridGNN
- **Best for**: Capturing diverse graph properties
- **Key features**: Combines GCN, GIN, and GAT layers, hierarchical pooling
- **Parameters**: Standard only

### 4. PlanarGNN
- **Best for**: Specifically designed for planar graphs
- **Key features**: Planar-aware convolutions, skip connections, attention readout
- **Parameters**: Standard only

### 5. SimpleButEffectiveGNN
- **Best for**: Small graphs with limited data
- **Key features**: Minimal architecture, just 2 GIN layers
- **Parameters**: Standard only

## Usage

```python
from GNN_architectures import create_gnn_model

# Create a GIN model
model = create_gnn_model(
    architecture='gin',
    num_features=26,
    hidden_dim=64,
    num_classes=2,
    dropout=0.2,
    num_layers=3
)

# Create a GAT model
model = create_gnn_model(
    architecture='gat',
    num_features=26,
    hidden_dim=64,
    num_classes=2,
    dropout=0.2,
    num_heads=4,
    num_layers=2
)
```

## Architecture Selection Guide

| Architecture | Complexity | Parameters | Recommended for |
|-------------|------------|------------|-----------------|
| Simple | Low | ~10K | Quick testing, small datasets |
| GIN | Medium | ~50K | General purpose, good default |
| GAT | Medium-High | ~60K | Graphs with varying edge importance |
| Hybrid | High | ~100K | When unsure about best approach |
| Planar | Medium | ~40K | Specifically planar graphs |

## Key Parameters

- `num_features`: Input feature dimension (from GraphBuilder)
- `hidden_dim`: Hidden layer size (32-128 recommended)
- `num_classes`: Output classes (2 for binary classification)
- `dropout`: Regularization (0.1-0.3 recommended)

## Model Architecture Details

### GIN Architecture
```
Input → GIN layers (with batch norm) → Jumping Knowledge → Global Pooling → Classifier
```

### GAT Architecture
```
Input → Multi-head Attention layers → Global Attention Pooling → Classifier
```

## Tips
- Start with GIN for general tasks
- Use 2-3 layers for small graphs (avoid oversmoothing)
- SimpleButEffectiveGNN often performs best on small datasets
- Consider ensemble of different architectures for best results