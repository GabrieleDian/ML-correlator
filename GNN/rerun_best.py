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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--train_loop", nargs="*")
    parser.add_argument("--test_loop", nargs="*")
    parser.add_argument("--slurm", action="store_true")
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

    # Folder for this rerun batch (scripts + logs): reruns/<project>/<sweep_id>/
    reruns_dir = Path("reruns") / args.project / sweep_id
    reruns_dir.mkdir(parents=True, exist_ok=True)

    slurm_logs_dir = reruns_dir / "slurm_logs"
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)

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

            slurm_path = reruns_dir / f"rerun_{idx}.sbatch"
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

    print("\n🎉 DONE — All reruns launched successfully.")


if __name__ == "__main__":
    main()
