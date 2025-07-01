"""
Example script showing how to run hyperparameter sweeps for GNN models.
"""

import torch
import numpy as np
import pandas as pd
import ast
from pathlib import Path

# Import the sweep utilities
from sweep_utils import (
    run_sweep, 
    run_sweep_from_config, 
    quick_sweep,
    analyze_sweep_results,
    create_example_config_file
)
from GraphBuilder_with_features import create_graph_dataset


def load_graph_data(loop):
    """Load graph data from CSV files."""
    edges = []
    y = []
    
    filename = f'../Graph_Edge_Data/den_graph_data_{loop}.csv'
    df = pd.read_csv(filename)
    edges += df['EDGES'].tolist()
    y += df['COEFFICIENTS'].tolist()
    
    edges = [ast.literal_eval(e) for e in edges]
    graphs_data = list(zip(edges, y))
    return graphs_data


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load your data
    print("Loading graph data...")
    graphs_data = load_graph_data(loop=8)
    
    # 2. Create dataset with your chosen features
    print("Creating dataset...")
    feature_config = {
        'selected_features': ['basic', 'face', 'spectral_node', 'centrality'],
        'laplacian_pe_k': 3
    }
    
    dataset, scaler = create_graph_dataset(graphs_data, feature_config)
    print(f"Dataset created with {len(dataset)} graphs")
    print(f"Feature dimensions: {dataset[0].x.shape[1]}")
    
    # 3. Define hyperparameter ranges for grid search
    param_ranges = {
        'hidden_channels': [32, 64, 128],
        'num_layers': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.003, 0.01],
        'weight_decay': [0, 1e-4, 5e-4]
    }
    
    # Fixed configuration (not swept)
    fixed_config = {
        'model_name': 'gin',
        'epochs': 100,
        'batch_size': 32,
        'scheduler_type': 'onecycle',
        'save_models': False  # Set to True if you want to save all models
    }
    
    # Calculate total number of runs
    total_runs = 1
    for param, values in param_ranges.items():
        total_runs *= len(values)
    print(f"\nTotal number of grid search combinations: {total_runs}")
    
    # 4. Run the sweep
    project_name = "gnn-planar-graphs-sweep"
    sweep_name = "hyperparameter_optimization"
    
    print(f"\nStarting sweep '{sweep_name}' in project '{project_name}'...")
    sweep_id = run_sweep(
        param_ranges=param_ranges,
        dataset=dataset,
        project_name=project_name,
        fixed_config=fixed_config,
        sweep_name=sweep_name,
        count=None  # None means run all combinations for grid search
    )
    
    print(f"\nSweep completed! Sweep ID: {sweep_id}")
    
    # 5. Analyze results
    print("\nAnalyzing sweep results...")
    results = analyze_sweep_results(project_name, sweep_id)
    
    # Print best configuration
    if results['best_config']:
        print("\nBest Configuration:")
        print(f"Validation Accuracy: {results['best_config']['best_val_accuracy']:.4f}")
        print("Hyperparameters:")
        for param, value in results['best_config']['config'].items():
            if param in param_ranges:
                print(f"  {param}: {value}")
        
        # Print top 5 configurations
        print("\nTop 5 Configurations:")
        for i, config in enumerate(results['all_results'][:5]):
            print(f"\n{i+1}. Val Acc: {config['best_val_accuracy']:.4f}")
            print("   Config:", {k: v for k, v in config['config'].items() if k in param_ranges})
    
    # Save results
    results_path = f"sweep_results_{sweep_id}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


def run_quick_sweep_example():
    """Example using the quick_sweep function for a smaller search."""
    # Load data and create dataset
    graphs_data = load_graph_data(loop=8)
    dataset, _ = create_graph_dataset(
        graphs_data,
        {'selected_features': ['basic', 'face', 'spectral_node'], 'laplacian_pe_k': 3}
    )
    
    # Run a quick sweep with fewer hyperparameter combinations
    sweep_id = quick_sweep(
        dataset=dataset,
        project_name="gnn-planar-graphs-quick",
        hidden_channels=[32, 64],
        num_layers=[2, 3],
        dropout=[0.1, 0.2],
        lr=[0.001, 0.003],
        weight_decay=[0, 1e-4],
        epochs=50,  # Fewer epochs for quick testing
        model_name='gin'
    )
    
    print(f"Quick sweep completed! Sweep ID: {sweep_id}")


def run_from_config_example():
    """Example using a configuration file."""
    # First, create an example config file
    create_example_config_file("my_sweep_config.yaml")
    
    # Load data and create dataset
    graphs_data = load_graph_data(loop=8)
    dataset, _ = create_graph_dataset(
        graphs_data,
        {'selected_features': ['basic', 'face', 'spectral_node'], 'laplacian_pe_k': 3}
    )
    
    # Run sweep from config file
    sweep_id = run_sweep_from_config(
        config_path="my_sweep_config.yaml",
        dataset=dataset,
        project_name="gnn-planar-graphs-config",
        count=None  # Run all combinations
    )
    
    print(f"Sweep from config completed! Sweep ID: {sweep_id}")


if __name__ == "__main__":
    # Choose which example to run
    print("GNN Hyperparameter Sweep Examples")
    print("1. Full sweep with all hyperparameters")
    print("2. Quick sweep with fewer combinations")
    print("3. Sweep from configuration file")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        run_quick_sweep_example()
    elif choice == "3":
        run_from_config_example()
    else:
        print("Invalid choice. Running full sweep by default.")
        main()
