# training_utils.py

## Overview
This module provides the core training functionality for GNN models, including training loops, evaluation, and metrics computation with WandB integration.

## Main Functions

### `train(config, dataset)`
Main training function that handles the complete training pipeline.

**Parameters:**
- `config`: SimpleNamespace object with training configuration
- `dataset`: List of PyTorch Geometric Data objects

**Returns:**
- Dictionary with:
  - `model_state`: Best model state dict
  - `final_train_acc`: Final training accuracy
  - `best_val_acc`: Best validation accuracy
  - `best_epoch`: Epoch with best validation accuracy

### `train_epoch(model, train_loader, optimizer, device, scheduler, scheduler_type)`
Executes one training epoch.

### `evaluate(model, val_loader, device)`
Evaluates model on validation set.

### `compute_metrics(y_true, y_pred)`
Computes classification metrics including precision, recall, F1, and confusion matrix.

## Configuration

Required configuration parameters:
```python
config = SimpleNamespace(
    # Model parameters
    model_name='gin',              # Architecture name
    in_channels=18,               # Input features
    hidden_channels=64,           # Hidden dimensions
    dropout=0.2,                  # Dropout rate
    num_layers=3,                 # Number of layers (optional)
    
    # Training parameters
    lr=0.001,                     # Learning rate
    weight_decay=5e-4,            # L2 regularization
    epochs=100,                   # Number of epochs
    batch_size=32,                # Batch size
    
    # Scheduler (optional)
    scheduler_type='plateau',     # 'plateau', 'onecycle', or None
    
    # WandB (optional)
    use_wandb=True,              # Enable WandB logging
    project='project-name',       # WandB project
    experiment_name='exp-1'       # Experiment name
)
```

## Supported Schedulers

### 1. OneCycleLR
```python
scheduler_type='onecycle'
# Automatically configured with:
# - max_lr = 3 * base_lr
# - 30% warmup
# - Cosine annealing
```

### 2. ReduceLROnPlateau
```python
scheduler_type='plateau'
# Reduces LR by 50% after 10 epochs without improvement
# Min LR: 1e-5
```

## Usage Example

```python
from training_utils import train
from types import SimpleNamespace

# Configure training
config = SimpleNamespace(
    model_name='gin',
    in_channels=dataset[0].x.shape[1],
    hidden_channels=64,
    num_layers=3,
    dropout=0.2,
    lr=0.001,
    weight_decay=5e-4,
    epochs=100,
    batch_size=32,
    scheduler_type='onecycle',
    use_wandb=True,
    project='gnn-planar-graphs',
    experiment_name='baseline'
)

# Train model
results = train(config, dataset)

print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
print(f"Achieved at epoch: {results['best_epoch']}")

# Save best model
torch.save(results['model_state'], 'best_model.pt')
```

## Features

1. **Automatic Train/Val Split**: 80/20 split with fixed seed for reproducibility
2. **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
3. **Best Model Tracking**: Automatically saves best model based on validation accuracy
4. **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1, and confusion matrix
5. **WandB Integration**: Optional automatic logging of all metrics

## Logged Metrics

When WandB is enabled, the following metrics are logged:
- `train_loss`, `train_accuracy`, `train_precision`, `train_recall`, `train_f1`
- `val_loss`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`
- `current_lr` (learning rate)
- Confusion matrices (printed every 10 epochs)

## Tips

1. **Learning Rate**: Start with 0.001 for OneCycleLR, 0.01 for plateau scheduler
2. **Batch Size**: 32 works well for graphs with 11-14 nodes
3. **Early Stopping**: Model automatically saves best validation checkpoint
4. **Debugging**: Set `use_wandb=False` to disable logging during debugging
5. **Scheduler Choice**: OneCycleLR often works better for fixed-length training