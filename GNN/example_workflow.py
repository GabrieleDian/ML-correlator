"""
Example workflow showing the complete process:
1. Compute features (once)
2. Train models using pre-computed features
"""

import subprocess
import sys
from pathlib import Path
import yaml


def step1_compute_features(config_path='config.yaml'):
    """Step 1: Compute all features based on config."""
    print("=" * 60)
    print("STEP 1: Computing features")
    print("=" * 60)
    
    # Load config to check loop order
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loop_order = config['data']['loop_order']
    base_dir = Path(config['data']['base_dir'])
    
    # Check if features already exist
    features_dir = base_dir / f'features_loop_{loop_order}'
    if features_dir.exists() and len(list(features_dir.glob('*.npy'))) > 0:
        print(f"Features already exist for loop {loop_order}")
        return
    
    # Compute all features using config
    cmd = [sys.executable, 'compute_features.py', '--config', config_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def step2_train_model(config_path='config.yaml'):
    """Step 2: Train model using config."""
    print("\n" + "=" * 60)
    print("STEP 2: Training model")
    print("=" * 60)
    
    # Train model using config
    cmd = [sys.executable, 'one_run_simple.py', '--config', config_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    """Run the complete workflow."""
    print("COMPLETE WORKFLOW EXAMPLE")
    print("This will:")
    print("1. Compute features based on config.yaml")
    print("2. Train a GNN model using those features")
    print()
    
    config_path = 'config.yaml'
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"ERROR: {config_path} not found!")
        print("Please create a config.yaml file first.")
        return
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Using configuration from {config_path}:")
    print(f"  Loop order: {config['data']['loop_order']}")
    print(f"  Features to compute: {config['features']['features_to_compute']}")
    print(f"  Features for training: {config['features']['selected_features']}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print()
    
    # Step 1: Compute features (only needs to be done once)
    step1_compute_features(config_path)
    
    # Step 2: Train model (can be done many times with different configs)
    step2_train_model(config_path)
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Modify config.yaml to try different settings")
    print("2. Run again with: python example_workflow.py")


if __name__ == "__main__":
    main()
