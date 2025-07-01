"""
Example workflow showing the complete process:
1. Compute features (once)
2. Train models using pre-computed features
"""

import subprocess
import sys
from pathlib import Path


def step1_compute_features(loop_order=8):
    """Step 1: Compute all features for a given loop order."""
    print("=" * 60)
    print("STEP 1: Computing features")
    print("=" * 60)
    
    # Check if features already exist
    features_dir = Path(f'Graph_Edge_Data/features_loop_{loop_order}')
    if features_dir.exists() and len(list(features_dir.glob('*.npy'))) > 0:
        print(f"Features already exist for loop {loop_order}")
        return
    
    # Compute all features
    cmd = [sys.executable, 'compute_features.py', '--loop', str(loop_order)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def step2_train_model(loop_order=8, features=['degree', 'betweenness']):
    """Step 2: Train model using pre-computed features."""
    print("\n" + "=" * 60)
    print("STEP 2: Training model")
    print("=" * 60)
    
    # Train model
    cmd = [
        sys.executable, 'one_run_simple.py',
        '--loop', str(loop_order),
        '--features'] + features + [
        '--epochs', '50'  # Use fewer epochs for demo
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    """Run the complete workflow."""
    print("COMPLETE WORKFLOW EXAMPLE")
    print("This will:")
    print("1. Compute features for loop 8 (if not already done)")
    print("2. Train a GNN model using those features")
    print()
    
    # You can change these parameters
    loop_order = 8
    features_to_use = ['degree', 'betweenness', 'clustering']
    
    # Step 1: Compute features (only needs to be done once)
    step1_compute_features(loop_order)
    
    # Step 2: Train model (can be done many times with different configs)
    step2_train_model(loop_order, features_to_use)
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Try different feature combinations")
    print("2. Try different model architectures")
    print("3. Add more features to compute_features.py")


if __name__ == "__main__":
    main()
