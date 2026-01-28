#!/usr/bin/env python3
import argparse
import yaml
import os
import copy
import subprocess
import wandb
from pathlib import Path

GNN_DIR = Path(__file__).resolve().parent


# Only these hparams are imported from the sweep config
ALLOWED_SWEEP_KEYS = {
    "learning_rate",
    "weight_decay",
    "dropout",
    "hidden_channels",
    "num_layers",
    "batch_size",
    "scheduler_type",
    "threshold",
    "model_name",
}


def merge_configs(base_cfg, sweep_cfg, epochs, project, run_name,
                  train_loop_override=None, test_loop_override=None):
    """Combine base_config + sweep hyperparameters into a clean final config."""

    cfg = copy.deepcopy(base_cfg)

    # -----------------------
    # Apply sweep hyperparameters
    # -----------------------
    for key, value in sweep_cfg.items():
        if key not in ALLOWED_SWEEP_KEYS:
            continue

        if key in ["learning_rate", "weight_decay", "batch_size",
                   "scheduler_type", "threshold"]:
            cfg.setdefault("training", {})[key] = value

        elif key in ["dropout", "hidden_channels", "num_layers", "model_name"]:
            cfg.setdefault("model", {})[key] = value

    # -----------------------
    # Override epochs
    # -----------------------
    cfg.setdefault("training", {})["epochs"] = epochs

    # -----------------------
    # Loop order overrides
    # -----------------------
    if train_loop_override:
        cfg.setdefault("data", {})["train_loop_order"] = train_loop_override
    if test_loop_override:
        cfg.setdefault("data", {})["test_loop_order"] = test_loop_override

    # -----------------------
    # Force features from base_config only
    # -----------------------
    cfg.setdefault("features", {})
    cfg["features"]["selected_features"] = base_cfg["features"]["selected_features"]

    # -----------------------
    # WandB info
    # -----------------------
    cfg.setdefault("experiment", {})
    cfg["experiment"]["use_wandb"] = True
    cfg["experiment"]["wandb_project"] = project
    cfg["experiment"]["wandb_name"] = f"rerun_{run_name}"

    return cfg


def _resolve_base_dir(base_dir_value: str | None, config_path: Path) -> Path:
    if not base_dir_value:
        raise RuntimeError("data.base_dir is missing in config.")
    p = Path(str(base_dir_value))
    if p.is_absolute():
        return p

    # Prefer repo root (..../ML-correlator)
    repo_root = Path(__file__).resolve().parents[1]
    cand = (repo_root / p).resolve()
    if cand.exists():
        return cand

    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand

    # Legacy: relative to config file
    return (config_path.parent / p).resolve()


def _normalize_loop_order(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, (str, int)):
        s = str(value)
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s.strip()]
    return [str(value)]


def _load_test_dataset_from_config(config_path: Path):
    """Load (concat) test dataset according to config.

    NOTE: This aims to match one_run_simple.py behavior: fit scaler on TRAIN and apply to test.
    """
    import torch
    from graph_builder import create_simple_dataset

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    features_cfg = cfg.get("features", {})

    selected_features = features_cfg.get("selected_features", None)
    if not selected_features:
        raise RuntimeError("features.selected_features is missing/empty in config.")

    base_dir = _resolve_base_dir(data_cfg.get("base_dir", None), config_path)

    train_exts = _normalize_loop_order(data_cfg.get("train_loop_order", None))
    test_exts = _normalize_loop_order(data_cfg.get("test_loop_order", None))

    if len(train_exts) == 0:
        raise RuntimeError("data.train_loop_order is missing/empty in config (needed to fit scaler).")
    if len(test_exts) == 0:
        raise RuntimeError("data.test_loop_order is missing/empty in config.")

    # Optional perf knobs (if present)
    n_jobs = int(data_cfg.get("n_jobs", 1))
    chunk_size = int(data_cfg.get("chunk_size", 100000))

    # Fit scaler on train loops
    train_scaler = None
    max_features = None
    for ext in train_exts:
        ds, scaler, feats = create_simple_dataset(
            file_ext=str(ext),
            selected_features=selected_features,
            normalize=True,
            data_dir=str(base_dir),
            scaler=train_scaler,
            max_features=max_features,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        if train_scaler is None:
            train_scaler = scaler
        if max_features is None or feats > max_features:
            max_features = feats

    # Load test loops using train scaler + max_features
    test_datasets = []
    for ext in test_exts:
        ds, _, _ = create_simple_dataset(
            file_ext=str(ext),
            selected_features=selected_features,
            normalize=True,
            data_dir=str(base_dir),
            scaler=train_scaler,
            max_features=max_features,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        test_datasets.append(ds)

    test_dataset = torch.utils.data.ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
    return test_dataset


def _predict_probs_from_checkpoint(config_path: Path, checkpoint_path: Path):
    """Return (y_true:int ndarray, y_prob:float ndarray) for the test set from one checkpoint."""
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader

    from GNN_architectures import create_gnn_model

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    test_dataset = _load_test_dataset_from_config(config_path)
    loader = DataLoader(test_dataset, batch_size=int(training_cfg.get("batch_size", 64)), shuffle=False)

    # Determine num_features
    first_graph = test_dataset[0] if not hasattr(test_dataset, "datasets") else test_dataset.datasets[0][0]
    num_features = int(first_graph.x.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_gnn_model(
        architecture=model_cfg.get("name", "gin"),
        num_features=num_features,
        hidden_dim=int(model_cfg.get("hidden_channels", 64)),
        num_classes=1,
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_layers=int(model_cfg.get("num_layers", 3)),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out.view(-1))
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(batch.y.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs).astype(float) if len(all_probs) else np.array([], dtype=float)
    y_true = np.concatenate(all_labels).astype(int) if len(all_labels) else np.array([], dtype=int)
    return y_true, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--train_loop", nargs="*")
    parser.add_argument("--test_loop", nargs="*")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument(
        "--average",
        action="store_true",
        help="After launching reruns, also compute per-sample mean/std probabilities across all best-tagged runs and save as a CSV under models/<project>/<sweep_id>/.",
    )
    args = parser.parse_args()

    # Load sweep
    api = wandb.Api()
    sweep = api.sweep(args.sweep_path)
    runs = [r for r in sweep.runs if "best" in (r.tags or [])]
    print(f"[INFO] Found {len(runs)} best-tagged runs.")

    # Load base config
    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    out_dir = Path(args.project)
    out_dir.mkdir(exist_ok=True)

    # Extract wandb sweep ID (last part of sweep_path)
    sweep_id = args.sweep_path.strip().split("/")[-1]

    # Folder for slurm logs: <project>/<sweep_id>/
    slurm_logs_dir = out_dir / sweep_id
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)

    # Track config paths for averaging
    generated_configs = []
    # Track checkpoint paths for averaging
    generated_checkpoints = []

    # -----------------------
    # Process runs
    # -----------------------
    for idx, run in enumerate(runs):
        print(f"\n=== Preparing rerun for: {run.name} ===")

        safe_name = run.name.replace(" ", "_")
        sweep_cfg = run.config or {}

        # Merge configs cleanly
        final_cfg = merge_configs(
            base_cfg=base_cfg,
            sweep_cfg=sweep_cfg,
            epochs=args.epochs,
            project=args.project,
            run_name=safe_name,
            train_loop_override=args.train_loop,
            test_loop_override=args.test_loop,
        )
        # --------------------------------------------------------
        # Create dedicated folder: models/<project>/<sweep_id>/
        # --------------------------------------------------------
        project_name = args.project
        run_name_clean = safe_name  # run.name already cleaned above
        save_folder = Path("models") / project_name / sweep_id
        save_folder.mkdir(parents=True, exist_ok=True)

        # Make sure experiment block exists
        final_cfg.setdefault("experiment", {})

        # Assign model save directory + filename
        final_cfg["experiment"]["model_dir"] = str(save_folder)
        final_cfg["experiment"]["model_name"] = run_name_clean
        final_cfg["experiment"]["wandb_name"] = f"rerun_{safe_name}"

        print(f"[INFO] Model will be saved to: {save_folder}/{run_name_clean}.pt")

        # ==========================
        # Save configs under configs/<project_name>/
        # ==========================
        project_configs_dir = Path("configs") / args.project
        project_configs_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = project_configs_dir / f"{safe_name}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(final_cfg, f)

        print(f"[INFO] Saved combined config to: {yaml_path}")

        generated_configs.append(yaml_path)
        generated_checkpoints.append(save_folder / f"{run_name_clean}.pt")

        print(f"[INFO] Saved: {yaml_path}")
        print("[DEBUG] features:", final_cfg["features"]["selected_features"])
        print("[DEBUG] epochs:", final_cfg["training"]["epochs"])

        # Slurm job
        if args.slurm:
            job_script = "\n".join([
                "#!/bin/bash",
                f"#SBATCH --job-name=rerun_{idx}",
                f"#SBATCH --output={slurm_logs_dir / f'slurm_rerun_{idx}.out'}",
                f"#SBATCH --error={slurm_logs_dir / f'slurm_rerun_{idx}.err'}",
                "#SBATCH --partition=maxgpu",
                "#SBATCH --time=1-00:00:00",
                "#SBATCH --cpus-per-task=70",
                "#SBATCH --mem=700G",
                "",
                f"cd {GNN_DIR}",
                f"python one_run_simple.py --config {yaml_path}",
                "",
            ])

            slurm_path = out_dir / f"rerun_{idx}.sbatch"
            with open(slurm_path, "w") as f:
                f.write(job_script)

            subprocess.run(["sbatch", str(slurm_path)])
            continue

        # Local run
        print(f"[INFO] Running locally...")
        subprocess.run(
            ["python", "one_run_simple.py", "--config", str(yaml_path)],
            cwd=GNN_DIR,
        )

    # --------------------------------------------------------
    # Optional: average predictions over all best-tagged runs
    # --------------------------------------------------------
    if args.average:
        if args.slurm:
            raise RuntimeError("--average is not supported with --slurm (jobs run asynchronously). Run without --slurm to aggregate immediately.")

        import numpy as np
        import pandas as pd

        if len(generated_configs) == 0:
            raise RuntimeError("No configs were generated; cannot average predictions.")

        # Folder for output CSV is the same as model_dir used above
        sweep_id = args.sweep_path.strip().split("/")[-1]
        save_folder = Path("models") / args.project / sweep_id
        save_folder.mkdir(parents=True, exist_ok=True)

        pred_probs = []
        y_true_ref = None

        for cfg_path, ckpt_path in zip(generated_configs, generated_checkpoints):
            if not ckpt_path.exists():
                raise RuntimeError(f"Checkpoint not found (did training save it?): {ckpt_path}")

            y_true, y_prob = _predict_probs_from_checkpoint(cfg_path, ckpt_path)

            if y_true_ref is None:
                y_true_ref = y_true
            else:
                if len(y_true) != len(y_true_ref) or not np.array_equal(y_true, y_true_ref):
                    raise RuntimeError(
                        "Predictions are not aligned across runs (different dataset ordering/length). "
                        "Ensure all runs evaluate the exact same test file(s) and ordering."
                    )

            pred_probs.append(y_prob)

        probs = np.stack(pred_probs, axis=0)  # [R, N]
        mean = probs.mean(axis=0)
        std = probs.std(axis=0, ddof=1) if probs.shape[0] > 1 else np.zeros_like(mean)

        out_df = pd.DataFrame({
            "index": np.arange(len(mean), dtype=int),
            "y_true": y_true_ref,
            "prob_mean": mean,
            "prob_std": std,
        })

        out_path = save_folder / "ensemble_test_predictions_mean_std.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved averaged predictions CSV → {out_path}")

    print("\nDONE — All reruns launched successfully.")


if __name__ == "__main__":
    main()
