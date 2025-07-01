# sweep_utils.py

## Overview
This module provides comprehensive hyperparameter optimization functionality using WandB Sweeps with grid search methodology.

## Main Functions

### `run_sweep(param_ranges, dataset, project_name, fixed_config, sweep_name, count)`
Execute a complete hyperparameter sweep.

**Parameters:**
- `param_ranges`: Dict of parameters to sweep with their values
- `dataset`: Preprocessed dataset
- `project_name`: WandB project name
- `fixed_config`: Parameters that remain constant
- `sweep_name`: Name for this sweep
- `count`: Max runs (None for grid search = all combinations)

**Returns:** Sweep ID

### `quick_sweep(dataset, project_name, **kwargs)`
Convenience function for quick hyperparameter sweeps with sensible defaults.

### `run_sweep_from_config(config_path, dataset, project_name, count)`
Run sweep from YAML/JSON configuration file.

### `analyze_sweep_results(project_name, sweep_id)`
Analyze completed sweep results.

**Returns:** Dictionary with:
- `best_config`: Best performing configuration
- `all_results`: All run results sorted by performance
- `statistics`: Summary statistics

## Usage Examples

### 1. Basic Grid Search
```python
from sweep_utils import run_sweep

param_ranges = {
    'hidden_channels': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.001, 0.003, 0.01],
    'weight_decay': [0, 1e-4, 5e-4]
}

sweep_id = run_sweep(
    param_ranges=param_ranges,
    dataset=dataset,
    project_name="gnn-optimization",
    fixed_config={'model_name': 'gin', 'epochs': 100}
)
```

### 2. Quick Sweep
```python
from sweep_utils import quick_sweep

sweep_id = quick_sweep(
    dataset=dataset,
    project_name="gnn-quick-test",
    hidden_channels=[32, 64],
    num_layers=[2, 3],
    epochs=50  # Override default
)
```

### 3. Configuration File
```python
from sweep_utils import run_sweep_from_config

# First create example config
create_example_config_file("my_sweep.yaml")

# Run from config
sweep_id = run_sweep_from_config(
    "my_sweep.yaml",
    dataset,
    "gnn-project"
)
```

### 4. Analyze Results
```python
from sweep_utils import analyze_sweep_results

results = analyze_sweep_results("gnn-project", sweep_id)

# Best configuration
print(f"Best accuracy: {results['best_config']['best_val_accuracy']:.4f}")
print(f"Best params: {results['best_config']['config']}")

# Statistics
stats = results['statistics']
print(f"Mean accuracy: {stats['mean_val_accuracy']:.4f} ± {stats['std_val_accuracy']:.4f}")
```

## Configuration File Format

Example YAML configuration:
```yaml
sweep_name: hyperparameter_optimization

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
  scheduler_type: plateau
  save_models: false
```

## Sweep Configuration Options

### Parameter Types
```python
# List of values (grid search)
param_ranges = {
    'hidden_channels': [32, 64, 128]
}

# Single value
param_ranges = {
    'batch_size': 32
}

# Advanced distributions (for future random/bayesian search)
param_ranges = {
    'lr': {'distribution': 'log_uniform', 'min': 1e-4, 'max': 1e-1}
}
```

### Fixed Configuration
Common fixed parameters:
- `model_name`: Architecture to use
- `epochs`: Training epochs
- `batch_size`: Batch size
- `scheduler_type`: LR scheduler
- `save_models`: Whether to save all models

## Helper Functions

### `create_example_config_file(filepath)`
Generates template configuration file.

### `train_sweep_iteration(dataset, fixed_config, feature_names)`
Single training iteration called by WandB agent (internal use).

## Best Practices

1. **Start Small**: Test with reduced parameter ranges first
2. **Monitor Progress**: Use WandB dashboard to track runs
3. **Resource Management**: Calculate total combinations before starting
4. **Iterative Refinement**: Use initial results to narrow search space
5. **Parallel Execution**: Run multiple agents for faster completion

## Grid Search Combinations

Calculate total runs:
```python
total = 1
for values in param_ranges.values():
    total *= len(values)
print(f"Total combinations: {total}")
```

## WandB Integration

The module automatically:
- Creates sweep configuration
- Initializes WandB runs
- Logs all metrics and parameters
- Tracks best performing models
- Provides sweep visualizations

## Tips

1. **Reduce Search Space**: Start with 2-3 values per parameter
2. **Use Count Parameter**: Limit runs for testing: `count=10`
3. **Save Models Selectively**: Enable `save_models` only for final sweeps
4. **Check Dashboard**: Monitor progress at wandb.ai
5. **Parallel Agents**: Run multiple terminals with same sweep_id for parallel execution