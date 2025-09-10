# Replace your main() function with this safer version:
import torch
import numpy as np
from types import SimpleNamespace
import argparse
from pathlib import Path
import yaml
import wandb
import os

# Import the simple dataset creator
from graph_builder import create_simple_dataset, quick_dataset_stats
from training_utils import train
from types import SimpleNamespace

def parse_loop_orders(value):
    """Parse loop orders - accepts single int or comma-separated list"""
    if ',' in value:
        # Multiple values: "7,8,9"
        return [int(x.strip()) for x in value.split(',')]
    else:
        # Single value: "7"
        return int(value)

def config_to_namespace(config_dict):
    """Convert nested config dict to SimpleNamespace for compatibility."""
    return SimpleNamespace(
        # Data configuration
        base_dir=config_dict.get('data', {}).get('base_dir', '../Graph_Edge_Data'),
        train_loop_order=config_dict.get('data', {}).get('train_loop_order', None),
        test_loop_order=config_dict.get('data', {}).get('test_loop_order', None),
        extra_train=config_dict.get('data', {}).get('extra_train', False),

        # Model configuration
        model_name=config_dict.get('model', {}).get('name', 'gin'),
        hidden_channels=config_dict.get('model', {}).get('hidden_channels', 64),
        num_layers=config_dict.get('model', {}).get('num_layers', 3),
        dropout=config_dict.get('model', {}).get('dropout', 0.2),
        
        # Training configuration
        lr=config_dict.get('training', {}).get('learning_rate', 0.0005),
        weight_decay=config_dict.get('training', {}).get('weight_decay', 0.001),
        epochs=config_dict.get('training', {}).get('epochs', 100),
        batch_size=config_dict.get('training', {}).get('batch_size', 32),
        scheduler_type=config_dict.get('training', {}).get('scheduler_type', 'onecycle'),
        threshold=config_dict.get('training', {}).get('threshold', 0.5),
        log_threshold_curves=config_dict.get('training', {}).get('log_threshold_curves', False),  # NEW
        
        # WandB configuration
        use_wandb=config_dict.get('experiment', {}).get('use_wandb', True),
        project=config_dict.get('experiment', {}).get('wandb_project', 'cluster-7-loop'),
        experiment_name=config_dict.get('experiment', {}).get('wandb_name', 'gin_simple'),
    )


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
        if hasattr(sweep_config, 'train_loop') and 'data' in config_dict:
            config_dict['data']['train_loop_order'] = sweep_config.train_loop
        if hasattr(sweep_config, 'test_loop') and 'data' in config_dict:
            config_dict['data']['test_loop_order'] = sweep_config.test_loop
        if hasattr(sweep_config, 'epochs') and 'training' in config_dict:
            config_dict['training']['epochs'] = sweep_config.epochs
        if hasattr(sweep_config, 'threshold') and 'training' in config_dict:
            config_dict['training']['threshold'] = sweep_config.threshold
            
        print(f"Sweep parameters: {dict(sweep_config)}")
        
    else:
        print("Running normal command line mode")
        


        # Normal run - parse command line arguments
        parser = argparse.ArgumentParser(description='Train GNN with pre-computed features')
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
        parser.add_argument('--train_loop', type=parse_loop_orders, 
                   help='Train Loop order(s) - single int (7) or comma-separated list (7,8,9)')
        parser.add_argument('--test_loop', type=parse_loop_orders, default=None,
                   help='Test Loop order(s) - single int or comma-separated list')
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
        if args.train_loop is not None:
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['train_loop_order'] = args.train_loop
        
        if args.test_loop is not None:
            if 'data' not in config_dict:
                config_dict['data'] = {}
            config_dict['data']['test_loop_order'] = args.test_loop
            
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
    train_loop_order = config_dict.get('data', {}).get('train_loop_order', 7)
    test_loop_order = config_dict.get('data', {}).get('test_loop_order', 8)
    extra_train = config_dict.get('data', {}).get('extra_train', False)
    selected_features = config_dict.get('features', {}).get('selected_features', ['degree'])
    seed = config_dict.get('experiment', {}).get('seed', 42)
    base_dir = config_dict.get('data', {}).get('base_dir', 'Graph_Edge_Data')  # Use config path with fallback    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Training on loop orders {train_loop_order}, testing on loop orders {test_loop_order} with features: {selected_features}")
    
        # Create training dataset
    if extra_train:
        # Load both "extra" and "normal" training sets
        train_dataset_extra, train_scaler, max_features = create_simple_dataset(
            loop_order=train_loop_order,
            selected_features=selected_features,
            normalize=True,
            data_dir=base_dir,
            extra_train=True
        )
        train_dataset_normal, _, _ = create_simple_dataset(
            loop_order=train_loop_order,
            selected_features=selected_features,
            normalize=True,
            scaler=train_scaler,       # keep same scaler
            max_features=max_features, # keep same dimensions
            data_dir=base_dir,
            extra_train=False
        )

        # Combine both into final training dataset
        train_dataset = train_dataset_extra + train_dataset_normal
        print(f"Combined training dataset: {len(train_dataset)} graphs "
            f"(extra: {len(train_dataset_extra)}, normal: {len(train_dataset_normal)})")

    else:
        # Normal behavior
        train_dataset, train_scaler, max_features = create_simple_dataset(
            loop_order=train_loop_order,
            selected_features=selected_features,
            normalize=True,
            data_dir=base_dir,
            extra_train=False
        )

    # Handle test dataset
    if test_loop_order is None or test_loop_order == train_loop_order:
        # Do 80-20 split on training data
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Train size: {train_size}, Test size: {test_size}")
    else:
        # Use separate test data
        test_dataset, _, _ = create_simple_dataset(
            loop_order=test_loop_order,
            selected_features=selected_features,
            normalize=True,
            scaler=train_scaler,       # reuse training scaler
            max_features=max_features, # enforce same dims
            data_dir=base_dir,
            extra_train=False          # tests never use "extra"
        )

    
    # Print dataset statistics
    quick_dataset_stats(train_dataset)
    
    # Convert config to SimpleNamespace for compatibility
    config = config_to_namespace(config_dict)
    config.in_channels = train_dataset[0].x.shape[1]
    
    # Update experiment name
    config.experiment_name = f"{config.model_name}_train_loop_{train_loop_order}_,test_loop_{test_loop_order}_train_{'_'.join(selected_features)}"
    
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Input features: {config.in_channels}")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Epochs: {config.epochs}")
    
    # Train model
    print("\nStarting training...")
    results = train(config, train_dataset, test_dataset)
    
    # Print results
    print("\nTraining completed!")
    print(f"Best test accuracy: {results['best_test_acc']:.4f} at epoch {results['best_epoch']}")
    print(f"Final training accuracy: {results['final_train_acc']:.4f}")
    print(f"Final training ROC AUC: {results['final_train_roc_auc']:.4f}")
    print(f"Final training PR AUC: {results['final_train_pr_auc']:.4f}")
    print(f"Final test ROC AUC: {results['final_test_roc_auc']:.4f}")
    print(f"Final test PR AUC: {results['final_test_pr_auc']:.4f}")
    
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