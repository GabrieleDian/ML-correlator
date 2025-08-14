# Simple Feature Pre-computation System

This is a minimal system for pre-computing graph features to speed up GNN training.

## Quick Start

### 1. Create Configuration File

Create a `config.yaml` file (example provided) with your settings.

### 2. Compute Features (Do this ONCE)

```bash
# Compute all features using config
python compute_features.py --config config.yaml

# Or override loop from command line
python compute_features.py --config config.yaml --loop 8

# Or compute just one feature
python compute_features.py --config config.yaml --feature degree
```

### 3. Train Model

```bash
# Train using config file, specify train and test loop orders
python one_run_simple.py --config config.yaml --train_loop [7,8] test_loop 9

# Override specific parameters
python one_run_simple.py --config config.yaml --train_loop 8 test_loop 8 --epochs 200
python one_run_simple.py --config config.yaml --features degree betweenness
```

### 4. Complete Example

```bash
# Run the complete workflow
python example_workflow.py
```

## File Structure

```
your_project/
├── compute_features.py      # Compute and save features
├── load_features.py        # Load saved features
├── GraphBuilder_simple.py  # Simple dataset builder
├── one_run_simple.py      # Training script
├── example_workflow.py    # Complete example
└── Graph_Edge_Data/
    ├── den_graph_data_8.csv
    └── features_loop_8/
        ├── degree.npy
        ├── betweenness.npy
        └── clustering.npy
```

## Available Features
- `identity_columns` : Columns of the identity matrix
- `adjacency_columns` : Columns of the adjacency matrix
- `eigen_1` : Eigenvector of the highest eigenvalue
- `eigen_2` : Eigenvector of the second highest eigenvalue
- `eigen_3` : Eigenvector of the third highest eigenvalue
- `degree`: Node degree
- `betweenness`: Betweenness centrality
- `clustering`: Clustering coefficient
- `closeness`: Closeness centrality
- `pagerank`: PageRank scores
- `face_count`: Number of faces (planar graphs)

## Adding New Features

1. Add a function in `compute_features.py`:
```python
def compute_my_feature(graphs_batch):
    features = []
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        # Compute your feature
        my_feature = [...]  # List of values for each node
        features.append(my_feature)
    return features
```

2. Register it in `FEATURE_FUNCTIONS`:
```python
FEATURE_FUNCTIONS = {
    ...
    'my_feature': compute_my_feature
}
```

3. Compute it:
```bash
python compute_features.py --loop 8 --feature my_feature
```

## Tips

1. **Start Small**: Test with one feature first
2. **Check Features**: Use `load_features.py` to verify saved features
3. **Parallel Processing**: Use `--n-jobs 4` to speed up computation
4. **Memory Issues**: Reduce `--chunk-size` if you run out of memory

## Common Issues

**"Feature not found"**: Run `compute_features.py` first

**Out of memory**: Reduce chunk size: `--chunk-size 500`

**Slow computation**: Increase parallel jobs: `--n-jobs 8`
