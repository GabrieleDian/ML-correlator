"""
Simplified training script that uses pre-computed features.
Much faster than the original version.
"""

import torch
import numpy as np
from types import SimpleNamespace
import argparse
from pathlib import Path
import yaml
import wandb
import os

# Import the simple dataset creator
from mixed_loop_builder import create_simple_dataset, quick_dataset_stats
from training_utils_mixed import train

def main():
    """Main function to run training with pre-computed features."""
    
    # Check if running as part of wandb sweep
    if 'WANDB_SWEEP_ID' in os.environ:
        print("Running as part of wandb sweep")
        
        # Initialize wandb for sweep
        wandb.init()
        sweep_config = wandb.config
        
        # Load base config file (from sweep parameters)
        config_file = sweep_config.get('config', 'config.yaml')
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Override with sweep parameters
        if hasattr(sweep_config, 'hidden_channels'):
            config_dict['model']['hidden_channels'] = sweep_config.hidden_channels
        if hasattr(sweep_config, 'num_layers'):
            config_dict['model']['num_layers'] = sweep_config.num_layers
        if hasattr(sweep_config, 'dropout'):
            config_dict['model']['dropout'] = sweep_config.dropout
        if hasattr(sweep_config, 'learning_rate'):
            config_dict['training']['learning_rate'] = sweep_config.learning_rate
        if hasattr(sweep_config, 'batch_size'):
            config_dict['training']['batch_size'] = sweep_config.batch_size
        if hasattr(sweep_config, 'features'):
            config_dict['features']['selected_features'] = sweep_config.features
        if hasattr(sweep_config, 'loop'):
            config_dict['data']['loop_order'] = sweep_config.loop
        if hasattr(sweep_config, 'epochs'):
            config_dict['training']['epochs'] = sweep_config.epochs
            
        print(f"Sweep parameters: {dict(sweep_config)}")
        
    else:
        print("Running normal command line mode")
        
        # Normal run - parse command line arguments
        parser = argparse.ArgumentParser(description='Train GNN with pre-computed features')
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
        parser.add_argument('--loop', type=int, help='Loop order (overrides config)')
        parser.add_argument('--features', nargs='+', help='Features to use (overrides config)')
        parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
        parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
        
        args = parser.parse_args()
        
        # Load config
        with open(args.config,'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Apply command line overrides
        if args.loop is not None:
            config_dict['data']['loop_order'] = args.loop
        if args.features is not None:
            config_dict['features']['selected_features'] = args.features
        if args.epochs is not None:
            config_dict['training']['epochs'] = args.epochs
        if args.seed is not None:
            config_dict['experiment']['seed'] = args.seed

def config_to_namespace(config_dict):
    """Convert nested config dict to SimpleNamespace for compatibility."""
    return SimpleNamespace(
        # Model configuration
        model_name=config_dict['model']['name'],
        hidden_channels=config_dict['model']['hidden_channels'],
        num_layers=config_dict['model']['num_layers'],
        dropout=config_dict['model']['dropout'],
        
        # Training configuration
        lr=config_dict['training']['learning_rate'],
        weight_decay=config_dict['training']['weight_decay'],
        epochs=config_dict['training']['epochs'],
        batch_size=config_dict['training']['batch_size'],
        scheduler_type=config_dict['training']['scheduler_type'],
        
        # WandB configuration
        use_wandb=config_dict['experiment']['use_wandb'],
        project=config_dict['experiment']['wandb_project'],
        experiment_name=config_dict['experiment']['wandb_name'],
    )


def main():
    """Main function to run training with pre-computed features."""
    parser = argparse.ArgumentParser(description='Train GNN with pre-computed features')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--loop', type=int, help='Loop order (overrides config)')
    parser.add_argument('--sweep', action='store_true', help='Run as wandb sweep agent')  # ADD THIS LINE
    parser.add_argument('--features', nargs='+', help='Features to use (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    
    args = parser.parse_args()
   
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.loop is not None:
        config_dict['data']['loop_order'] = args.loop
    if args.features is not None:
        config_dict['features']['selected_features'] = args.features
    if args.epochs is not None:
        config_dict['training']['epochs'] = args.epochs
    if args.seed is not None:
        config_dict['experiment']['seed'] = args.seed
    
    # Extract parameters
    loop_order = config_dict['data']['loop_order']
    selected_features = config_dict['features']['selected_features']
    seed = config_dict['experiment']['seed']
    base_dir = config_dict['data']['base_dir']
    
     # Fixed loop orders
    train_loops = [7, 8, 9]
    val_loop = 10

    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Training on loop {loop_order} with features: {selected_features}")
    
    # Create dataset with pre-computed features
    print("\nLoading pre-computed features...")
    # Pass base_dir to dataset creation
    train_dataset, scaler = create_simple_dataset(
        loop_order=[7,8,9],
        selected_features=selected_features,
        normalize=True,
        data_dir=base_dir
    )
    val_dataset, scaler = create_simple_dataset(
        loop_order=10,
        selected_features=selected_features,
        normalize=True,
        data_dir=base_dir
    )
    # Print dataset statistics
    quick_dataset_stats(train_dataset)
    
    # Convert config to SimpleNamespace for compatibility
    config = config_to_namespace(config_dict)
    config.in_channels = train_dataset[0].x.shape[1]
    
    # Update experiment name
    config.experiment_name = f"{config.model_name}_loop{loop_order}_{'_'.join(selected_features)}"
    
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Input features: {config.in_channels}")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Epochs: {config.epochs}")
    
    # Train model
    print("\nStarting training...")
    results = train(config, train_dataset, val_dataset)
    
    # Print results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f} at epoch {results['best_epoch']}")
    print(f"Final training accuracy: {results['final_train_acc']:.4f}")
    
    # Save model if requested
    if config_dict['experiment']['save_model']:
        save_dir = Path(config_dict['experiment']['model_dir'])
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f"{config.experiment_name}_best.pt"
        torch.save(results['model_state'], model_path)
        print(f"Model saved to: {model_path}")


def train_with_sweep():
    """Training function called by wandb sweep agent."""
    # Initialize wandb run (gets config from sweep)
    wandb.init()
    
    # Get sweep parameters
    sweep_config = wandb.config
    
    # Load your base config file
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override config with sweep parameters
    if hasattr(sweep_config, 'hidden_channels'):
        config_dict['model']['hidden_channels'] = sweep_config.hidden_channels
    if hasattr(sweep_config, 'num_layers'):
        config_dict['model']['num_layers'] = sweep_config.num_layers
    if hasattr(sweep_config, 'dropout'):
        config_dict['model']['dropout'] = sweep_config.dropout
    if hasattr(sweep_config, 'lr'):
        config_dict['training']['learning_rate'] = sweep_config.lr
    if hasattr(sweep_config, 'weight_decay'):
        config_dict['training']['weight_decay'] = sweep_config.weight_decay
    if hasattr(sweep_config, 'batch_size'):
        config_dict['training']['batch_size'] = sweep_config.batch_size
    if hasattr(sweep_config, 'features'):
        config_dict['features']['selected_features'] = sweep_config.features
    
    # Extract parameters (same as your existing code)
    loop_order = config_dict['data']['loop_order']
    selected_features = config_dict['features']['selected_features']
    seed = config_dict['experiment']['seed']
    base_dir = config_dict['data']['base_dir']
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Sweep run: loop {loop_order}, features: {selected_features}")
    
    train_dataset, scaler = create_simple_dataset(
        loop_order=[7,8,9],
        selected_features=selected_features,
        normalize=True,
        data_dir=base_dir
    )
    val_dataset, scaler = create_simple_dataset(
        loop_order=10,
        selected_features=selected_features,
        normalize=True,
        data_dir=base_dir
    )
    
    # Convert config to namespace (same as your existing code)
    config = config_to_namespace(config_dict)
    config.in_channels = dataset[0].x.shape[1]
    
    # Update experiment name for sweep
    config.experiment_name = f"sweep_{config.model_name}_loop{loop_order}_{'_'.join(selected_features)}"
    
    # Train model (same as your existing code)
    results = train(config, train_dataset, val_dataset)
    
    # Log final results to wandb (this is automatic with wandb.init())
    print(f"Sweep run completed - Val Acc: {results['best_val_acc']:.4f}")
    
    wandb.finish()
if __name__ == "__main__":
    main()
