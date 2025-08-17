import os
import wandb

# === CONFIGURE THESE ===
LOCAL_WANDB_FOLDER = "./wandb"  # Folder containing your runs and sweep folders
PROJECT_NAME = "ML-correlator-GNN_old"
ENTITY_NAME = "aliajrigers-desydeutsches-elektronen-synchrotron"
TARGET_DATE = "20250811"  # Only restore runs from this date

created_sweeps = {}
wandb.login()

def restore_run(run_path, sweep_id=None):
    run_name = os.path.basename(run_path)
    print(f"Restoring run: {run_name}")

    config = {}
    config_file = os.path.join(run_path, "config.yaml")
    if os.path.exists(config_file):
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

    sweep_tag = None
    if sweep_id:
        if sweep_id not in created_sweeps:
            sweep_config = {
                'name': f'restored-{sweep_id}',
                'method': 'grid',
                'parameters': {k: {'value': v} for k, v in config.items()} if config else {}
            }
            sweep_tag = wandb.sweep(sweep_config, project=PROJECT_NAME, entity=ENTITY_NAME)
            created_sweeps[sweep_id] = sweep_tag
            print(f"✅ Created new sweep: restored-{sweep_id}")
        else:
            sweep_tag = created_sweeps[sweep_id]

    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        config=config,
        reinit=True,
        name=run_name,
        group=f"restored-{sweep_id}" if sweep_id else None
    )

    files_dir = os.path.join(run_path, "files")
    if os.path.exists(files_dir):
        for f in os.listdir(files_dir):
            run.log_artifact(wandb.Artifact(name=f, type='dataset'))

    run.finish()
    print(f"✅ Run {run_name} restored.\n")

def restore_folder(local_folder, target_date):
    if not os.path.exists(local_folder):
        print(f"Folder {local_folder} does not exist!")
        return

    items = [d for d in os.listdir(local_folder) if os.path.isdir(os.path.join(local_folder, d))]
    print(f"Found {len(items)} folders. Starting restore for date {target_date}...\n")

    for item in items:
        item_path = os.path.join(local_folder, item)
        if item.startswith("run-") and item[4:12] == target_date:
            restore_run(item_path)
        elif item.startswith("sweep-"):
            sweep_id = item.split("-")[1]
            run_dirs = [os.path.join(item_path, d) for d in os.listdir(item_path)
                        if os.path.isdir(os.path.join(item_path, d)) and d.startswith("run-") and d[4:12] == target_date]
            if run_dirs:
                print(f"Restoring sweep {item} with {len(run_dirs)} runs for {target_date}.")
                for run_dir in run_dirs:
                    restore_run(run_dir, sweep_id=sweep_id)
        else:
            continue  # skip unknown folders or runs with different dates

    print("All runs from target date processed.")

if __name__ == "__main__":
    restore_folder(LOCAL_WANDB_FOLDER, TARGET_DATE)
