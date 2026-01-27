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
from graph_builder import create_simple_dataset, print_dataset_stats
from types import SimpleNamespace
from training_utils import train

# Handle properly list of values for training and testing.
import ast

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
        pos_weight=config_dict.get('training', {}).get('pos_weight', None),

        # WandB configuration
        use_wandb=config_dict.get('experiment', {}).get('use_wandb', True),
        project=config_dict.get('experiment', {}).get('wandb_project', 'cluster-7-loop'),
        experiment_name=config_dict.get('experiment', {}).get('wandb_name', 'gin_simple'),
    )


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
            ('training', 'pos_weight'),
            ('features', 'selected_features'),
            ('data', 'train_loop_order'),
            ('data', 'test_loop_order'),
            ('training', 'epochs'),
            ('training', 'threshold'),
        ]:
            if hasattr(sweep_config, key) and section in config_dict:
                config_dict[section][key] = getattr(sweep_config, key)
            # override model_name if present in sweep
            if hasattr(sweep_config, "model_name"):
                config_dict.setdefault("model", {})["name"] = getattr(sweep_config, "model_name")

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

    # --- Auto-tune data loading parameters based on node specs ---
    import psutil

    n_cpus = psutil.cpu_count(logical=True)
    mem_gb = psutil.virtual_memory().total / 1e9

    # Use ~75 % of CPUs for n_jobs
    auto_n_jobs = int(0.25 * n_cpus)

    # Scale chunk_size based on available memory
    if mem_gb < 128:
        auto_chunk_size = 70000
    elif mem_gb < 256:
        auto_chunk_size = 120000
    elif mem_gb < 512:
        auto_chunk_size = 250000
    elif mem_gb < 768:
        auto_chunk_size = 500000
    else:
        auto_chunk_size = 1000000

    config_dict.setdefault("data", {})
    config_dict["data"]["n_jobs"] = auto_n_jobs
    config_dict["data"]["chunk_size"] = auto_chunk_size

    print(f"[INFO] Auto-set n_jobs={auto_n_jobs}, chunk_size={auto_chunk_size} for {mem_gb:.1f} GB RAM")

    # ---------------------------------------------------------
    # 3. Data preparation (configurable validation split)
    train_loop_orders = normalize_loop_order(config_dict.get('data', {}).get('train_loop_order', '7'))
    test_loop_orders = normalize_loop_order(config_dict.get('data', {}).get('test_loop_order', '8'))
    selected_features = config_dict.get('features', {}).get('selected_features', ['degree'])
    seed = config_dict.get('experiment', {}).get('seed', 42)
    base_dir = config_dict.get('data', {}).get('base_dir', 'Graph_Edge_Data')
    use_val = config_dict.get('data', {}).get('val', False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Training on loop orders {train_loop_orders}, testing on loop orders {test_loop_orders} with features: {selected_features}")
    print(f"Validation enabled: {use_val}")

    # --- Load datasets ---
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
            max_features=max_features,
            n_jobs=auto_n_jobs,
            chunk_size=auto_chunk_size
        )
        train_datasets.append(ds)

    if train_scaler is None:
        train_scaler = scaler
    if max_features is None or feats > max_features:
        max_features = feats

    val_dataset = None

    if use_val:
        last_train_ds = train_datasets[-1]
        n_total = len(last_train_ds)
        n_val = int(0.30 * n_total)
        n_discard = n_total - n_val

        discard_ds, val_ds = torch.utils.data.random_split(
            last_train_ds,
            [n_discard, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        val_dataset = val_ds
        train_datasets = train_datasets[:-1]

        final_max_features = max(d.x.shape[1] for ds in train_datasets for d in ds)
        for ds in train_datasets:
            for data in ds:
                n_nodes, n_feats = data.x.shape
                if n_feats < final_max_features:
                    pad = torch.zeros(n_nodes, final_max_features - n_feats)
                    data.x = torch.cat([data.x, pad], dim=1)

        train_dataset = torch.utils.data.ConcatDataset(train_datasets) if train_datasets else None

    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        final_max_features = max(d.x.shape[1] for ds in train_datasets for d in ds)

    # Test set (skipped when validation enabled)
    if use_val:
        test_dataset = None
    else:
        test_datasets = []
        for file_ext in test_loop_orders:
            ds, _, _ = create_simple_dataset(
                file_ext=file_ext,
                selected_features=selected_features,
                normalize=True,
                scaler=train_scaler,
                max_features=final_max_features,
                data_dir=base_dir,
                n_jobs=auto_n_jobs,
                chunk_size=auto_chunk_size
            )
            test_datasets.append(ds)

        test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    # Convert config to SimpleNamespace for compatibility
    config = config_to_namespace(config_dict)

    # Determine global max feature width across ALL datasets
    import itertools

    def iter_graphs(ds):
        if ds is None:
            return []
        if isinstance(ds, torch.utils.data.ConcatDataset):
            return itertools.chain.from_iterable(ds.datasets)
        return ds

    all_datasets = [train_dataset]
    if val_dataset is not None:
        all_datasets.append(val_dataset)
    if test_dataset is not None:
        all_datasets.append(test_dataset)

    global_in_channels = max(g.x.shape[1] for ds in all_datasets for g in iter_graphs(ds))
    config.in_channels = global_in_channels

    # Print dataset info
    print_dataset_stats(train_dataset, "Train")
    if val_dataset is not None:
        print_dataset_stats(val_dataset, "Validation")
    if test_dataset is not None:
        print_dataset_stats(test_dataset, "Test")

    # Train model
    print("\nStarting training...")
    results = train(config, train_dataset, val_dataset, test_dataset, use_wandb=use_wandb)

    # Print final results
    print("\nTraining completed!\n")

    def safe_fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and value is not None else "N/A"

    has_val = val_dataset is not None

    print("=== Accuracy ===")
    print(f"  Train: {safe_fmt(results.get('train_acc'))}")
    if has_val:
        print(f"  Val:   {safe_fmt(results.get('val_acc'))}")
    if test_dataset is not None:
        print(f"  Test:  {safe_fmt(results.get('test_acc'))}\n")
    else:
        print("  Test:  SKIPPED\n")

    print("=== PR AUC ===")
    print(f"  Train: {safe_fmt(results.get('train_pr_auc'))}")
    if has_val:
        print(f"  Val:   {safe_fmt(results.get('val_pr_auc'))}")
    if test_dataset is not None:
        print(f"  Test:  {safe_fmt(results.get('test_pr_auc'))}\n")
    else:
        print("  Test:  SKIPPED\n")

    print("=== ROC AUC ===")
    print(f"  Train: {safe_fmt(results.get('train_roc_auc'))}")
    if has_val:
        print(f"  Val:   {safe_fmt(results.get('val_roc_auc'))}")
    if test_dataset is not None:
        print(f"  Test:  {safe_fmt(results.get('test_roc_auc'))}\n")
    else:
        print("  Test:  SKIPPED\n")

    print("=== Recall ===")
    print(f"  Train: {safe_fmt(results.get('train_recall'))}")
    if has_val:
        print(f"  Val:   {safe_fmt(results.get('val_recall'))}")
    if test_dataset is not None:
        print(f"  Test:  {safe_fmt(results.get('test_recall'))}\n")
    else:
        print("  Test:  SKIPPED\n")

    print("=== True-negative rate (true 0s below min true-1 prob) ===")
    print(f"  Train: {safe_fmt(results.get('train_true_negative_rate'))}")
    if has_val:
        print(f"  Val:   {safe_fmt(results.get('val_true_negative_rate'))}")
    if test_dataset is not None:
        print(f"  Test:  {safe_fmt(results.get('test_true_negative_rate'))}\n")
    else:
        print("  Test:  SKIPPED\n")

    print("=== Min prob among true-1 graphs (lowest_prob_true1) ===")
    print(f"  Train: {safe_fmt(results.get('train_lowest_prob_true1'))}")
    if has_val:
        print(f"  Val:   {safe_fmt(results.get('val_lowest_prob_true1'))}")
    if test_dataset is not None:
        print(f"  Test:  {safe_fmt(results.get('test_lowest_prob_true1'))}\n")
    else:
        print("  Test:  SKIPPED\n")

    # Final WandB logging
    if use_wandb and wandb.run is not None:
        if results.get("number_of_parameters") is not None:
            wandb.log({"number_of_parameters": results.get("number_of_parameters")})

        if test_dataset is not None:
            wandb.log({
                "test_loss": results.get("test_loss", 0),
                "test_acc": results.get("test_acc", 0),
                "test_pr_auc": results.get("test_pr_auc", 0),
                "test_roc_auc": results.get("test_roc_auc", 0),
                "test_recall": results.get("test_recall", 0),
                "test_true_negative_rate": results.get("test_true_negative_rate", 0),
                "test_lowest_prob_true1": results.get("test_lowest_prob_true1"),
                "total_time": results.get("total_time", 0)
            })

    # Save model if requested
    save_model = config_dict.get('experiment', {}).get('save_model', False)
    if save_model:
        exp_cfg = config_dict.get('experiment', {})
        model_dir = exp_cfg.get('model_dir', 'models')
        model_name = exp_cfg.get('model_name', None)

        save_dir = Path(model_dir)
        save_dir.mkdir(exist_ok=True)

        if model_name is not None:
            filename = f"{model_name}.pt"
        else:
            filename = f"{config.experiment_name}_best.pt"

        model_path = save_dir / filename
        torch.save(results['model_state'], model_path)
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()