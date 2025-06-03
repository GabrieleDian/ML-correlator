"""
WandB Sweep utilities for hyperparameter optimization of GNN models.
Supports grid search over model hyperparameters.
"""

import wandb
import torch
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Callable
import yaml
import json
from pathlib import Path

from training_utils import train
from GraphBuilder_with_features import create_graph_dataset


def create_sweep_config(param_ranges: Dict[str, Any], 
                       project_name: str,
                       sweep_name: str = "gnn_sweep") -> Dict[str, Any]:
    """
    Create a WandB sweep configuration for grid search.
    
    Args:
        param_ranges: Dictionary containing parameter names and their ranges
            Example: {
                'hidden_channels': [32, 64, 128],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],
                'lr': [0.001, 0.003, 0.01],
                'weight_decay': [0, 1e-4, 5e-4]
            }
        project_name: WandB project name
        sweep_name: Name for this sweep
        
    Returns:
        Sweep configuration dictionary
    """
    sweep_config = {
        'method': 'grid',  # Grid search as requested
        'name': sweep_name,
        'metric': {
            'name': 'best_val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {}
    }
    
    # Convert parameter ranges to WandB format
    for param_name, param_values in param_ranges.items():
        if isinstance(param_values, list):
            sweep_config['parameters'][param_name] = {
                'values': param_values
            }
        elif isinstance(param_values, dict):
            # Support for other distribution types if needed later
            sweep_config['parameters'][param_name] = param_values
        else:
            # Single value
            sweep_config['parameters'][param_name] = {
                'value': param_values
            }
    
    return sweep_config


def train_sweep_iteration(dataset: List, 
                         fixed_config: Dict[str, Any],
                         feature_names: Optional[List[str]] = None):
    """
    Single training iteration for the sweep.
    This function will be called by the WandB agent.
    
    Args:
        dataset: The preprocessed dataset
        fixed_config: Fixed configuration parameters (not swept)
        feature_names: Optional list of feature names for logging
    """
    # Initialize wandb run
    wandb.init()
    
    # Get sweep parameters from wandb.config
    sweep_config = wandb.config
    
    # Create configuration combining fixed and swept parameters
    config = SimpleNamespace(
        # Model configuration (from sweep)
        model_name=fixed_config.get('model_name', 'gin'),
        hidden_channels=sweep_config.hidden_channels,
        num_layers=sweep_config.num_layers,
        dropout=sweep_config.dropout,
        
        # Training configuration (from sweep)
        lr=sweep_config.lr,
        weight_decay=sweep_config.weight_decay,
        
        # Fixed configuration
        epochs=fixed_config.get('epochs', 100),
        batch_size=fixed_config.get('batch_size', 32),
        scheduler_type=fixed_config.get('scheduler_type', 'plateau'),
        
        # WandB configuration
        use_wandb=True,  # Already initialized
        project=wandb.run.project,
        experiment_name=wandb.run.name,
        
        # Input channels (from dataset)
        in_channels=dataset[0].x.shape[1]
    )
    
    # Log configuration
    wandb.config.update({
        'model_name': config.model_name,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'scheduler_type': config.scheduler_type,
        'num_features': config.in_channels,
        'dataset_size': len(dataset)
    })
    
    if feature_names:
        wandb.config.update({'feature_names': feature_names})
    
    # Train the model
    results = train(config, dataset)
    
    # Log final results
    wandb.log({
        'best_val_accuracy': results['best_val_acc'],
        'final_train_accuracy': results['final_train_acc'],
        'best_epoch': results['best_epoch']
    })
    
    # Optionally save the best model
    if fixed_config.get('save_models', False):
        model_path = f"models/{wandb.run.name}_best_model.pt"
        torch.save(results['model_state'], model_path)
        wandb.save(model_path)
    
    wandb.finish()


def run_sweep(param_ranges: Dict[str, Any],
              dataset: List,
              project_name: str,
              fixed_config: Optional[Dict[str, Any]] = None,
              sweep_name: str = "gnn_hyperparameter_sweep",
              count: Optional[int] = None) -> str:
    """
    Run a complete hyperparameter sweep.
    
    Args:
        param_ranges: Dictionary of parameters to sweep over
        dataset: The preprocessed dataset
        project_name: WandB project name
        fixed_config: Fixed configuration parameters
        sweep_name: Name for this sweep
        count: Maximum number of runs (None for grid search means all combinations)
        
    Returns:
        Sweep ID
    """
    if fixed_config is None:
        fixed_config = {}
    
    # Create sweep configuration
    sweep_config = create_sweep_config(param_ranges, project_name, sweep_name)
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    # Create training function with dataset and fixed config bound
    def train_fn():
        train_sweep_iteration(dataset, fixed_config, 
                            feature_names=getattr(dataset[0], 'feature_names', None))
    
    # Run sweep agent
    wandb.agent(sweep_id, train_fn, count=count, project=project_name)
    
    return sweep_id


def run_sweep_from_config(config_path: str,
                         dataset: List,
                         project_name: str,
                         count: Optional[int] = None) -> str:
    """
    Run a sweep from a configuration file.
    
    Args:
        config_path: Path to YAML or JSON configuration file
        dataset: The preprocessed dataset
        project_name: WandB project name
        count: Maximum number of runs
        
    Returns:
        Sweep ID
    """
    # Load configuration
    config_path = Path(config_path)
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Extract parameter ranges and fixed config
    param_ranges = config.get('param_ranges', {})
    fixed_config = config.get('fixed_config', {})
    sweep_name = config.get('sweep_name', 'gnn_sweep')
    
    return run_sweep(param_ranges, dataset, project_name, fixed_config, sweep_name, count)


def analyze_sweep_results(project_name: str, sweep_id: str) -> Dict[str, Any]:
    """
    Analyze the results of a completed sweep.
    
    Args:
        project_name: WandB project name
        sweep_id: Sweep ID
        
    Returns:
        Dictionary containing sweep analysis
    """
    api = wandb.Api()
    sweep = api.sweep(f"{api.default_entity}/{project_name}/{sweep_id}")
    
    # Get all runs from the sweep
    runs = sweep.runs
    
    # Extract results
    results = []
    for run in runs:
        if run.state == "finished":
            result = {
                'name': run.name,
                'config': dict(run.config),
                'best_val_accuracy': run.summary.get('best_val_accuracy', 0),
                'final_train_accuracy': run.summary.get('final_train_accuracy', 0),
                'best_epoch': run.summary.get('best_epoch', 0)
            }
            results.append(result)
    
    # Sort by validation accuracy
    results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
    
    # Find best configuration
    if results:
        best_config = results[0]
        
        # Calculate statistics
        val_accuracies = [r['best_val_accuracy'] for r in results]
        
        analysis = {
            'best_config': best_config,
            'all_results': results,
            'statistics': {
                'best_val_accuracy': max(val_accuracies),
                'worst_val_accuracy': min(val_accuracies),
                'mean_val_accuracy': np.mean(val_accuracies),
                'std_val_accuracy': np.std(val_accuracies),
                'num_runs': len(results)
            }
        }
    else:
        analysis = {
            'best_config': None,
            'all_results': [],
            'statistics': {}
        }
    
    return analysis


# Example usage functions
def create_example_config_file(filepath: str = "sweep_config.yaml"):
    """
    Create an example configuration file for sweeps.
    """
    example_config = {
        'sweep_name': 'gnn_hyperparameter_optimization',
        'param_ranges': {
            'hidden_channels': [32, 64, 128],
            'num_layers': [2, 3, 4],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.001, 0.003, 0.01],
            'weight_decay': [0, 0.0001, 0.0005]
        },
        'fixed_config': {
            'model_name': 'gin',
            'epochs': 100,
            'batch_size': 32,
            'scheduler_type': 'plateau',
            'save_models': False
        }
    }
    
    with open(filepath, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    print(f"Example configuration saved to {filepath}")


def quick_sweep(dataset: List,
                project_name: str,
                hidden_channels: List[int] = [32, 64],
                num_layers: List[int] = [2, 3],
                dropout: List[float] = [0.1, 0.2],
                lr: List[float] = [0.001, 0.01],
                weight_decay: List[float] = [0, 0.0005],
                **kwargs) -> str:
    """
    Quick function to run a sweep with common hyperparameters.
    
    Args:
        dataset: The preprocessed dataset
        project_name: WandB project name
        hidden_channels: List of hidden channel sizes to try
        num_layers: List of number of layers to try
        dropout: List of dropout rates to try
        lr: List of learning rates to try
        weight_decay: List of weight decay values to try
        **kwargs: Additional fixed configuration parameters
        
    Returns:
        Sweep ID
    """
    param_ranges = {
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay
    }
    
    fixed_config = {
        'model_name': kwargs.get('model_name', 'gin'),
        'epochs': kwargs.get('epochs', 100),
        'batch_size': kwargs.get('batch_size', 32),
        'scheduler_type': kwargs.get('scheduler_type', 'plateau')
    }
    
    return run_sweep(param_ranges, dataset, project_name, fixed_config)


if __name__ == "__main__":
    # Create example configuration file
    create_example_config_file()
    
    print("\nExample usage:")
    print("1. Load your dataset using create_graph_dataset()")
    print("2. Run sweep: sweep_id = run_sweep_from_config('sweep_config.yaml', dataset, 'your-project')")
    print("3. Analyze results: results = analyze_sweep_results('your-project', sweep_id)")
