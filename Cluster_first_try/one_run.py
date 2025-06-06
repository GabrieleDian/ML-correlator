"""
Main script for training GNN models on planar graphs.
Converted from Claude_main_v2.ipynb
"""

# Import necessary libraries
import torch
import numpy as np
from types import SimpleNamespace
import argparse

# Import custom modules
from GraphBuilder_with_features import create_graph_dataset, load_graph_data, get_feature_configs
from training_utils import train

# Define the path to the data directory
from pathlib import Path

loop = 8
data_path = Path(f'Graph_Edge_Data/den_graph_data_{loop}.csv').resolve()

# Standard Configurations 
feat_conf = get_feature_configs()['full'] # configuration with all features

# OneCycle configuration
config = SimpleNamespace(
    # Model configuration
    model_name='gin',
    hidden_channels=64,
    num_layers=3,
    dropout=0.2,
    
    # Training configuration
    lr=0.003,  # Reasonable for OneCycleLR
    weight_decay=5e-4,
    epochs=100,
    batch_size=32,
    scheduler_type='onecycle',
    
    # WandB configuration
    use_wandb=True,
    project='cluster_first_try',
    experiment_name='gin_8_loop_gpu',
)





def run_single_experiment(config, dataset):
    """Run a single experiment with specified configuration."""
    config.in_channels = dataset[0].x.shape[1]
    
    print(f"Dataset created with {len(dataset)} graphs")
    print(f"Feature dimensions: {config.in_channels}")
    print(f"Feature names: {dataset[0].feature_names}")
    
    # Train model
    results = train(config, dataset)
    
    return results



def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Train GNN models on planar graphs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading graph data at loop {loop}...")
    graphs_data = load_graph_data(loop=loop, path = data_path)
    print(f"Loaded {len(graphs_data)} graphs")
    
    # Get feature configuration
    print(f"\nUsing feature configuration: {feat_conf}")
    print(f"Selected features: {feat_conf['selected_features']}")
    print(f"Laplacian PE k: {feat_conf['laplacian_pe_k']}")
    
    # Create dataset
    print("\nComputing and adding feature to dataset...")
    dataset_config = SimpleNamespace(
        selected_features=feat_conf['selected_features'],
        laplacian_pe_k=feat_conf['laplacian_pe_k']
    )
    
    dataset, scaler = create_graph_dataset(
        graphs_data,
        {
            'selected_features': dataset_config.selected_features,
            'laplacian_pe_k': dataset_config.laplacian_pe_k
        }
    )

 
    # Run experiment
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Scheduler: {config.scheduler_type}")
    
    results = run_single_experiment(config, dataset)
    
    # Print results
    print("\nExperiment completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f} at epoch {results['best_epoch']}")
    print(f"Final training accuracy: {results['final_train_acc']:.4f}")
    
    # Save model if requested
    #if hasattr(config, 'save_model') and config.save_model:
    #    model_path = f"models/{config.experiment_name}_best_model.pt"
    #    torch.save(results['model_state'], model_path)
    #    print(f"Model saved to: {model_path}")
    
    #return results


if __name__ == "__main__":
    main()