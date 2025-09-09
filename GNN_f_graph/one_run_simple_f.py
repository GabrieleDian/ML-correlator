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
from f_graph_builder import create_simple_dataset, quick_dataset_stats
from training_utils_f import train
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
        
        # WandB configuration
        use_wandb=config_dict.get('experiment', {}).get('use_wandb', True),
        project=config_dict.get('experiment', {}).get('wandb_project', 'cluster-7-loop'),
        experiment_name=config_dict.get('experiment', {}).get('wandb_name', 'gin_simple'),
    )

# Main function
def main():
    """Main function for regression on pre-computed features with two edge types."""

    # --- Load config ---
    parser = argparse.ArgumentParser(description='GNN regression with features')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--train_loop', type=parse_loop_orders)
    parser.add_argument('--test_loop', type=parse_loop_orders)
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Apply overrides safely
    if args.train_loop: config_dict.setdefault('data', {})['train_loop_order'] = args.train_loop
    if args.test_loop: config_dict.setdefault('data', {})['test_loop_order'] = args.test_loop
    if args.features: config_dict.setdefault('features', {})['selected_features'] = args.features
    if args.epochs: config_dict.setdefault('training', {})['epochs'] = args.epochs
    if args.seed: config_dict.setdefault('experiment', {})['seed'] = args.seed

    train_loop_order = config_dict['data']['train_loop_order']
    test_loop_order = config_dict['data']['test_loop_order']
    selected_features = config_dict['features']['selected_features']
    seed = config_dict['experiment'].get('seed', 42)

    # --- Set seeds ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Create datasets ---
    train_dataset, train_scaler, max_features = create_simple_dataset(
        loop_order=train_loop_order,
        selected_features=selected_features,
        normalize=True,
        data_dir=config_dict['data'].get('base_dir', 'Graph_Edge_Data')
    )

    if test_loop_order == train_loop_order:
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        test_dataset, _, _ = create_simple_dataset(
            loop_order=test_loop_order,
            selected_features=selected_features,
            normalize=True,
            scaler=train_scaler,
            max_features=max_features,
            data_dir=config_dict['data'].get('base_dir', 'Graph_Edge_Data')
        )

    quick_dataset_stats(train_dataset)

    # --- Convert config ---
    config = config_to_namespace(config_dict)
    config.in_channels = train_dataset[0].x.shape[1]

    # --- Train regression model ---
    results = train(config, train_dataset, test_dataset)

    # --- Print regression metrics ---
    print("Training completed!")
    print(f"Best test RMSE: {results['best_test_rmse']:.4f} at epoch {results['best_epoch']}")
    print(f"Final train RMSE: {results['final_train_rmse']:.4f}")
    print(f"Final test R²: {results['final_test_r2']:.4f}")

    # --- Save model if requested ---
    if config_dict.get('experiment', {}).get('save_model', False):
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