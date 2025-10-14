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


"""Convert nested config dict to SimpleNamespace for compatibility."""
def config_to_namespace(config_dict):
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
        learning_rate=config_dict.get('training', {}).get('learning_rate', 0.0005),
        weight_decay=config_dict.get('training', {}).get('weight_decay', 0.001),
        epochs=config_dict.get('training', {}).get('epochs', 100),
        batch_size=config_dict.get('training', {}).get('batch_size', 32),
        scheduler_type=config_dict.get('training', {}).get('scheduler_type', 'onecycle'),
        threshold=config_dict.get('training', {}).get('threshold', 0.5),
        log_threshold_curves=config_dict.get('training', {}).get('log_threshold_curves', False),

        # WandB configuration
        use_wandb=config_dict.get('experiment', {}).get('use_wandb', True),
        project=config_dict.get('experiment', {}).get('wandb_project', 'cluster-7-loop'),
        experiment_name=config_dict.get('experiment', {}).get('wandb_name', 'gin_simple'),
    )


# Handle properly list of values for training and testing.
import ast
def normalize_loop_order(value):
    """
    Normalize loop order input into a list of strings.
    Accepts: str ("7", "7to8", "7,8,9", "['7','8']")
            or list
    Returns: list[str]
    """
    if isinstance(value, list):
        return [str(v) for v in value]

    if isinstance(value, str):
        # Try to parse a Python list string
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        # Handle comma-separated case
        if "," in value:
            return [v.strip() for v in value.split(",")]
        return [value.strip()]
    raise ValueError(f"Unsupported loop order format: {value}")

def main():
    print("\n=== Starting GNN training run ===")

    is_sweep = 'WANDB_SWEEP_ID' in os.environ
    config_dict = {}

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    if is_sweep:
        print("[INFO] Running as part of a WandB sweep")

        # Initialize first, so wandb.config is populated
        wandb.init()
        sweep_config = wandb.config

        # Get project name (safe default if missing)
        project_from_sweep = getattr(sweep_config, "project", "cluster-7-loop")

        # Optional: you can also let sweep define a name
        name_from_sweep = getattr(sweep_config, "wandb_name", None)

        config_file = sweep_config.get("config", "config.yaml")
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Inject project info into runtime config
        config_dict.setdefault("experiment", {})["wandb_project"] = project_from_sweep
        if name_from_sweep:
            config_dict["experiment"]["wandb_name"] = name_from_sweep

        raw_use_wandb = True



        # Override sweep parameters
        for section, key in [
            ('model', 'hidden_channels'),
            ('model', 'num_layers'),
            ('model', 'dropout'),
            ('training', 'learning_rate'),
            ('training', 'batch_size'),
            ('features', 'selected_features'),
            ('data', 'train_loop_order'),
            ('data', 'test_loop_order'),
            ('training', 'epochs'),
            ('training', 'threshold'),
        ]:
            if hasattr(sweep_config, key) and section in config_dict:
                config_dict[section][key] = getattr(sweep_config, key)

        raw_use_wandb = True

    else:
        print(" Running in normal mode")
        parser = argparse.ArgumentParser(description='Train GNN with pre-computed features')
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
        parser.add_argument('--train_loop', type=str)
        parser.add_argument('--test_loop', type=str, default=None)
        parser.add_argument('--features', nargs='+')
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--seed', type=int)
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # CLI overrides
        if args.train_loop is not None:
            config_dict.setdefault('data', {})['train_loop_order'] = args.train_loop
        if args.test_loop is not None:
            config_dict.setdefault('data', {})['test_loop_order'] = args.test_loop
        if args.features is not None:
            config_dict.setdefault('features', {})['selected_features'] = args.features
        if args.epochs is not None:
            config_dict.setdefault('training', {})['epochs'] = args.epochs
        if args.seed is not None:
            config_dict.setdefault('experiment', {})['seed'] = args.seed

        raw_use_wandb = config_dict.get('experiment', {}).get('use_wandb', False)

    # ---------------------------------------------------------
    # 2. WandB initialization (for both sweeps and single runs)
    # ---------------------------------------------------------
    use_wandb = is_sweep or str(raw_use_wandb).lower() in ['1', 'true', 'yes']


    if use_wandb:
        if is_sweep:
            print(" WandB initialized automatically by sweep")
        else:
            print("Initializing WandB for single run...")
            wandb.init(
                project=config_dict.get('experiment', {}).get('wandb_project', 'cluster-7-loop'),
                config=config_dict,
                reinit=True
            )
    else:
        print(" WandB disabled in config.")


    # ---------------------------------------------------------
    # 3. Data preparation
    # Extract parameters with safe defaults
    train_loop_orders = normalize_loop_order(config_dict.get('data', {}).get('train_loop_order', '7'))
    test_loop_orders = normalize_loop_order(config_dict.get('data', {}).get('test_loop_order', '8'))
    selected_features = config_dict.get('features', {}).get('selected_features', ['degree'])
    seed = config_dict.get('experiment', {}).get('seed', 42)
    base_dir = config_dict.get('data', {}).get('base_dir', 'Graph_Edge_Data')  # Use config path with fallback
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Training on loop orders {train_loop_orders}, testing on loop orders {test_loop_orders} with features: {selected_features}")
    
        # Build training dataset
    train_datasets = []
    train_scaler = None
    max_features = None

    for file_ext in train_loop_orders:
        ds, scaler, feats = create_simple_dataset(
            file_ext=file_ext,
            selected_features=selected_features,
            normalize=True,
            data_dir=base_dir,
            scaler=train_scaler,
            max_features=max_features
        )
        train_datasets.append(ds)

        # Capture scaler/max_features from the first dataset
        if train_scaler is None:
            train_scaler = scaler
        if max_features is None:
            max_features = feats

    # Concatenate into a single training dataset
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Build test dataset
    if test_loop_orders == train_loop_orders:
        # Do 80-20 split on training data
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        test_datasets = []
        for file_ext in test_loop_orders:
            ds, _, _ = create_simple_dataset(
                file_ext=file_ext,
                selected_features=selected_features,
                normalize=True,
                scaler=train_scaler,
                max_features=max_features,
                data_dir=base_dir
            )
            test_datasets.append(ds)

        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    # Print dataset statistics
    quick_dataset_stats(train_dataset)
    
    # Convert config to SimpleNamespace for compatibility
    config = config_to_namespace(config_dict)
    config.in_channels = train_dataset[0].x.shape[1]
    
    # Update experiment name
    config.experiment_name = f"{config.model_name}_train_loop_{train_loop_orders}_,test_loop_{test_loop_orders}_train_{'_'.join(selected_features)}"
    
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Input features: {config.in_channels}")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Epochs: {config.epochs}")
    

    # Train model
    print("\nStarting training...")
    results = train(config, train_dataset, test_dataset, use_wandb=use_wandb)
    
    # Print results
    print("\nTraining completed!")
    print(f"Final test accuracy: {results['final_test_acc']:.4f}")
    print(f"Final training accuracy: {results['final_train_acc']:.4f}")
    print(f"Final training ROC AUC: {results['final_train_roc_auc']:.4f}")
    print(f"Final training PR AUC: {results['final_train_pr_auc']:.4f}")
    print(f"Final training recall: {results['final_train_recall']:.4f}")
    print(f"Final test ROC AUC: {results['final_test_roc_auc']:.4f}")
    print(f"Final test PR AUC: {results['final_test_pr_auc']:.4f}")
    print(f"Final test recall: {results['final_test_recall']:.4f}")

    # Final WandB logging and cleanup
    import contextlib
    import io
    if use_wandb and wandb.run is not None:
        print(" Logging final metrics to WandB...")
        wandb.log({
            "final_test_acc": results.get("final_test_acc", 0),
            "final_train_acc": results.get("final_train_acc", 0),
            "final_test_roc_auc": results.get("final_test_roc_auc", 0),
            "final_train_roc_auc": results.get("final_train_roc_auc", 0),
            "final_test_pr_auc": results.get("final_test_pr_auc", 0),
            "final_train_pr_auc": results.get("final_train_pr_auc", 0),
            "final_test_recall": results.get("final_test_recall", 0),
            "final_train_recall": results.get("final_train_recall", 0),
        })
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            wandb.finish()
    else:
        print(" WandB active flag mismatch — skipping final logging.")
        if wandb.run is None:
            print(" wandb.run is None; likely reinit=False or wandb disabled mid-run.")



    # Save model if requested
    save_model = config_dict.get('experiment', {}).get('save_model', False)
    if save_model:
        model_dir = config_dict.get('experiment', {}).get('model_dir', 'models')
        save_dir = Path(model_dir)
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f"{config.experiment_name}_best.pt"
        torch.save(results['model_state'], model_path)
        print(f"Model saved to: {model_path}")
    


if __name__ == "__main__":
    main()