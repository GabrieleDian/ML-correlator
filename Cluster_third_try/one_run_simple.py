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

# Import the simple dataset creator
from graphbuilder_simple import create_simple_dataset, quick_dataset_stats
from training_utils import train


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
    dataset, scaler = create_simple_dataset(
        loop_order=loop_order,
        selected_features=selected_features,
        normalize=True,
        data_dir=base_dir
    )
    
    # Print dataset statistics
    quick_dataset_stats(dataset)
    
    # Convert config to SimpleNamespace for compatibility
    config = config_to_namespace(config_dict)
    config.in_channels = dataset[0].x.shape[1]
    
    # Update experiment name
    config.experiment_name = f"{config.model_name}_loop{loop_order}_{'_'.join(selected_features)}"
    
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Input features: {config.in_channels}")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Epochs: {config.epochs}")
    
    # Train model
    print("\nStarting training...")
    results = train(config, dataset)
    
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


if __name__ == "__main__":
    main()
