"""
Simplified training script that uses pre-computed features.
Much faster than the original version.
"""

import torch
import numpy as np
from types import SimpleNamespace
import argparse

# Import the simple dataset creator
from graphbuilder_simple import create_simple_dataset, quick_dataset_stats
from training_utils import train


# Simple configuration
def get_simple_config():
    """Get a simple training configuration."""
    return SimpleNamespace(
        # Model configuration
        model_name='gin',
        hidden_channels=64,
        num_layers=3,
        dropout=0.2,
        
        # Training configuration
        lr=0.003,
        weight_decay=5e-4,
        epochs=100,
        batch_size=32,
        scheduler_type='onecycle',
        
        # WandB configuration (optional)
        use_wandb=False,  # Set to True if you want to use WandB
        project='planar_graphs',
        experiment_name='gin_simple',
    )


def main():
    """Main function to run training with pre-computed features."""
    parser = argparse.ArgumentParser(description='Train GNN with pre-computed features')
    parser.add_argument('--loop', type=int, default=8, help='Loop order')
    parser.add_argument('--features', nargs='+', 
                       default=['degree', 'betweenness', 'clustering'],
                       help='Features to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--use-wandb', action='store_true', help='Use WandB logging')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Training on loop {args.loop} with features: {args.features}")
    
    # Create dataset with pre-computed features
    print("\nLoading pre-computed features...")
    dataset, scaler = create_simple_dataset(
        loop_order=args.loop,
        selected_features=args.features,
        normalize=True
    )
    
    # Print dataset statistics
    quick_dataset_stats(dataset)
    
    # Get configuration
    config = get_simple_config()
    config.in_channels = dataset[0].x.shape[1]
    config.epochs = args.epochs
    config.use_wandb = args.use_wandb
    config.experiment_name = f'gin_loop{args.loop}_{"_".join(args.features)}'
    
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
    
    # Save model if needed
    #save_dir = Path('models')
    #save_dir.mkdir(exist_ok=True)
    #model_path = save_dir / f"{config.experiment_name}_best.pt"
    #torch.save(results['model_state'], model_path)
    #print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
