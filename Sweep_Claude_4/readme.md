# GNN Hyperparameter Optimization Pipeline

A comprehensive pipeline for hyperparameter optimization of Graph Neural Networks (GNNs) on planar graph classification tasks using Weights & Biases (WandB) Sweeps.

## Overview

This pipeline provides a systematic approach to finding optimal hyperparameters for GNN models through grid search. It's designed specifically for planar graphs with 11-14 nodes and supports multiple GNN architectures including GIN, GAT, and hybrid models.

## Features

- **Grid Search Optimization**: Systematically explores all hyperparameter combinations
- **Multiple GNN Architectures**: Support for GIN, GAT, Hybrid, Planar-specific, and Simple models
- **Rich Feature Engineering**: Specialized features for planar graphs including face-based, spectral, and dual graph features
- **WandB Integration**: Automatic experiment tracking and visualization
- **Flexible Configuration**: YAML/JSON configuration files or programmatic setup
- **Comprehensive Analysis**: Built-in tools for analyzing sweep results

## Repository Structure

```
.
├── GNN_architectures.py          # GNN model implementations
├── GraphBuilder_with_features.py # Feature extraction for planar graphs
├── training_utils.py             # Training loop and utilities
├── sweep_utils.py                # Main hyperparameter sweep module
├── run_sweep_example.py          # Example scripts for running sweeps
├── sweep_config.yaml             # Example configuration file
├── hyperparameter_sweep.ipynb    # Interactive Jupyter notebook
└── README.md                     # This file
```

## Installation

### Prerequisites

```bash
# Python 3.8+ required
pip install torch torch-geometric wandb numpy pandas scikit-learn networkx pyyaml
```

### Setup

1. Clone the repository
2. Install dependencies
3. Create a WandB account at https://wandb.ai
4. Login to WandB:
```bash
wandb login
```

## Quick Start

### 1. Basic Grid Search

```python
from sweep_utils import quick_sweep
from GraphBuilder_with_features import create_graph_dataset

# Load your data
graphs_data = load_graph_data(loop=8)

# Create dataset
dataset, scaler = create_graph_dataset(
    graphs_data,
    {'selected_features': ['basic', 'face', 'spectral_node', 'centrality'],
     'laplacian_pe_k': 3}
)

# Run quick sweep
sweep_id = quick_sweep(
    dataset=dataset,
    project_name="gnn-planar-graphs",
    hidden_channels=[32, 64],
    num_layers=[2, 3],
    dropout=[0.1, 0.2],
    lr=[0.001, 0.01],
    weight_decay=[0, 1e-4]
)
```

### 2. Using Configuration File

```bash
# Generate example config
python -c "from sweep_utils import create_example_config_file; create_example_config_file()"

# Edit sweep_config.yaml to your needs, then run:
```

```python
from sweep_utils import run_sweep_from_config

sweep_id = run_sweep_from_config(
    'sweep_config.yaml',
    dataset,
    'your-project-name'
)
```

### 3. Analyze Results

```python
from sweep_utils import analyze_sweep_results

results = analyze_sweep_results('your-project-name', sweep_id)
print(f"Best configuration: {results['best_config']}")
print(f"Best validation accuracy: {results['statistics']['best_val_accuracy']:.4f}")
```

## Hyperparameters

The pipeline optimizes the following hyperparameters:

| Parameter | Description | Default Range |
|-----------|-------------|---------------|
| `hidden_channels` | Hidden layer dimensions | [32, 64, 128] |
| `num_layers` | Number of GNN layers | [2, 3, 4] |
| `dropout` | Dropout rate | [0.1, 0.2, 0.3] |
| `lr` | Learning rate | [0.001, 0.003, 0.01] |
| `weight_decay` | L2 regularization | [0, 1e-4, 5e-4] |

## Advanced Usage

### Custom Parameter Ranges

```python
from sweep_utils import run_sweep

param_ranges = {
    'hidden_channels': [16, 32, 64, 128, 256],
    'num_layers': [1, 2, 3, 4, 5],
    'dropout': np.linspace(0, 0.5, 6).tolist(),
    'lr': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3]
}

fixed_config = {
    'model_name': 'gat',  # Try different architecture
    'epochs': 150,
    'batch_size': 64,
    'scheduler_type': 'onecycle'
}

sweep_id = run_sweep(
    param_ranges=param_ranges,
    dataset=dataset,
    project_name="gnn-advanced-sweep",
    fixed_config=fixed_config
)
```

### Feature Configuration

The pipeline supports various feature combinations for planar graphs:

```python
# Minimal features
feature_config = {
    'selected_features': ['basic', 'face'],
    'laplacian_pe_k': 0
}

# Full features
feature_config = {
    'selected_features': ['basic', 'face', 'spectral_node', 'dual', 'centrality'],
    'laplacian_pe_k': 4
}
```

Available feature groups:
- `basic`: Node degree
- `face`: Face-based features for planar graphs
- `spectral_node`: Node-level spectral features
- `spectral_global`: Graph-level spectral features
- `dual`: Dual graph properties
- `centrality`: Various centrality measures
- `laplacian_pe`: Laplacian positional encoding

### Model Architectures

Switch between different GNN architectures:

```python
fixed_config = {
    'model_name': 'gin',      # Options: 'gin', 'gat', 'hybrid', 'planar', 'simple'
    # ... other fixed params
}
```

## Configuration File Format

Example `sweep_config.yaml`:

```yaml
sweep_name: gnn_hyperparameter_optimization

param_ranges:
  hidden_channels: [32, 64, 128]
  num_layers: [2, 3, 4]
  dropout: [0.1, 0.2, 0.3]
  lr: [0.001, 0.003, 0.01]
  weight_decay: [0, 0.0001, 0.0005]

fixed_config:
  model_name: gin
  epochs: 100
  batch_size: 32
  scheduler_type: onecycle
  save_models: false
```

## Visualization and Analysis

The pipeline provides several analysis tools:

1. **Best Configuration Summary**
   ```python
   results = analyze_sweep_results(project_name, sweep_id)
   # Returns best config, all results, and statistics
   ```

2. **Jupyter Notebook Visualizations**
   - Hyperparameter importance plots
   - 2D heatmaps for parameter interactions
   - Distribution analysis

3. **WandB Dashboard**
   - Real-time training curves
   - Parallel coordinates plot
   - Hyperparameter importance

## Tips for Effective Sweeps

1. **Start Small**: Begin with a quick sweep using fewer parameter values
2. **Monitor Resources**: Grid search can be computationally expensive
3. **Use Early Stopping**: Set `count` parameter to limit runs
4. **Iterative Refinement**: Use results from initial sweeps to narrow ranges
5. **Feature Selection**: Experiment with different feature combinations

## Example Workflow

```python
# 1. Load data
graphs_data = load_graph_data(loop=8)

# 2. Create dataset with selected features
dataset, scaler = create_graph_dataset(
    graphs_data,
    {'selected_features': ['basic', 'face', 'spectral_node'],
     'laplacian_pe_k': 3}
)

# 3. Run quick test sweep
sweep_id = quick_sweep(
    dataset=dataset,
    project_name="gnn-test",
    epochs=50  # Fewer epochs for testing
)

# 4. Analyze results
results = analyze_sweep_results("gnn-test", sweep_id)

# 5. Run full sweep with refined ranges
param_ranges = {
    'hidden_channels': [48, 64, 96],  # Refined based on test
    'num_layers': [2, 3],
    'dropout': [0.15, 0.2, 0.25],
    'lr': [0.002, 0.003, 0.005],
    'weight_decay': [1e-4, 3e-4]
}

final_sweep_id = run_sweep(
    param_ranges=param_ranges,
    dataset=dataset,
    project_name="gnn-final",
    fixed_config={'epochs': 150}
)
```

## Troubleshooting

### Common Issues

1. **WandB Login**: Ensure you're logged in with `wandb login`
2. **Memory Issues**: Reduce batch size or model size
3. **Slow Training**: Consider using GPU or reducing parameter combinations
4. **Feature Extraction**: Some features require planar graphs

### Debug Mode

```python
# Run single configuration for debugging
config = SimpleNamespace(
    model_name='gin',
    hidden_channels=64,
    num_layers=3,
    dropout=0.2,
    lr=0.001,
    weight_decay=1e-4,
    epochs=10,  # Short run
    batch_size=32,
    use_wandb=False,  # Disable WandB for debugging
    in_channels=dataset[0].x.shape[1]
)

from training_utils import train
results = train(config, dataset)
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gnn_hyperopt_pipeline,
  title = {GNN Hyperparameter Optimization Pipeline},
  year = {2024},
  url = {https://github.com/yourusername/gnn-hyperopt}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.

## Support

For questions or issues:
1. Check the documentation in each module
2. Review example scripts and notebook
3. Open an issue on GitHub
4. Contact: your-email@example.com