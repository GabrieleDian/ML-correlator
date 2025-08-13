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
        parser.add_argument('--learning_rate', type=float)  
        
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
        
        print(f"Loaded config structure: {list(config_dict.keys())}")
        
        # Override with sweep parameters (with safe key access)
        if hasattr(sweep_config, 'hidden_channels') and 'model' in config_dict:
            config_dict['model']['hidden_channels'] = sweep_config.hidden_channels
        if hasattr(sweep_config, 'num_layers') and 'model' in config_dict:
            config_dict['model']['num_layers'] = sweep_config.num_layers
        if hasattr(sweep_config, 'dropout') and 'model' in config_dict:
            config_dict['model']['dropout'] = sweep_config.dropout
        if hasattr(sweep_config, 'learning_rate') and 'training' in config_dict:
            config_dict['training']['learning_rate'] = sweep_config.learning_rate
        if hasattr(sweep_config, 'batch_size') and 'training' in config_dict:
            config_dict['training']['batch_size'] = sweep_config.batch_size
        if hasattr(sweep_config, 'features') and 'features' in config_dict:
            config_dict['features']['selected_features'] = sweep_config.features
        if hasattr(sweep_config, 'loop') and 'data' in config_dict:
            config_dict['data']['loop_order'] = sweep_config.loop
        if hasattr(sweep_config, 'epochs') and 'training' in config_dict:
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
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Debug: print config structure
        print(f"Loaded config structure: {list(config_dict.keys())}")
        print(f"Config dict: {config_dict}")
        
        # Apply command line overrides with safe key access
        if args.loop is not None:
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['loop_order'] = args.loop
            
        if args.features is not None:
            if 'features' not in config_dict:
                config_dict['features'] = {}
            config_dict['features']['selected_features'] = args.features
            
        if args.epochs is not None:
            if 'training' not in config_dict:
                config_dict['training'] = {}
            config_dict['training']['epochs'] = args.epochs
            
        if args.seed is not None:
            if 'experiment' not in config_dict:
                config_dict['experiment'] = {}
            config_dict['experiment']['seed'] = args.seed
    
    # Extract parameters with safe defaults
    loop_order = config_dict.get('data', {}).get('loop_order', 7)
    selected_features = config_dict.get('features', {}).get('selected_features', ['degree'])
    seed = config_dict.get('experiment', {}).get('seed', 42)
    base_dir = config_dict.get('data', {}).get('base_dir', 'Graph_Edge_Data')
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
        
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
    results = train(config, train_dataset,val_dataset)
    
    # Print results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f} at epoch {results['best_epoch']}")
    print(f"Final training accuracy: {results['final_train_acc']:.4f}")
    
    # Save model if requested
    save_model = config_dict.get('experiment', {}).get('save_model', False)
    if save_model:
        model_dir = config_dict.get('experiment', {}).get('model_dir', 'models')
        save_dir = Path(model_dir)
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f"{config.experiment_name}_best.pt"
        torch.save(results['model_state'], model_path)
        print(f"Model saved to: {model_path}")
    
    # Close wandb if it was initialized for sweep
    if 'WANDB_SWEEP_ID' in os.environ:
        print("Finishing wandb sweep run")
        wandb.finish()


if __name__ == "__main__":
    main()