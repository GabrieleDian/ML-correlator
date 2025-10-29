"""
Simple functions to load pre-computed features.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import ast

data_dir = '../Graph_Edge_Data'


def load_saved_features(file_ext, feature_names, data_dir=data_dir):
    """
    Load pre-computed features for a given loop order.

    Args:
       file_ext: File extension (e.g., '7','7to8', 8)
        feature_names: List of feature names to load
        data_dir: Base directory for data
        extra_train: If True, load features from "features_loop_{loop_order}to"
                     instead of "features_loop_{loop_order}".

    Returns:
        features: Dict mapping feature names to numpy arrays
        labels: List of graph labels
    """
    # Select directory based on extra_train flag
    features_dir = Path(data_dir) / f'features_loop_{file_ext}'

    # Load labels (always from the same CSV)
    csv_path = Path(data_dir) / f'den_graph_data_{file_ext}.csv'
    df = pd.read_csv(csv_path)
    labels = df['COEFFICIENTS'].tolist()

    # Load requested features
    features = {}
    for feature_name in feature_names:
        feature_path = features_dir / f'{feature_name}.npy'
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature {feature_name} not found in {features_dir}. "
                f"Run compute_features.py first."
            )

        features[feature_name] = np.load(feature_path)
        print(f"Loaded {feature_name}: shape {features[feature_name].shape}")

    return features, labels



def get_available_features(loop_order, data_dir=data_dir):
    """Get list of available pre-computed features for a loop order."""
    features_dir = Path(data_dir) / f'features_loop_{loop_order}'
    
    if not features_dir.exists():
        return []
    
    # Check for .npy files
    feature_files = list(features_dir.glob('*.npy'))
    feature_names = [f.stem for f in feature_files]
    
    return sorted(feature_names)


import glob

def load_graph_structure(file_ext, data_dir=data_dir, extra_train=False):
    """
    Load original graph edges for creating edge_index.

    Args:
        file_ext: File extension (e.g., '7','7to8', 8)
        data_dir: Base directory
        extra_train: If True, load from all den_graph_data_{loop_order}to*.csv
                     instead of den_graph_data_{loop_order}.csv

    Returns:
        graph_infos: List of dicts with keys:
            - num_nodes
            - edge_list
            - node_labels
    """
    csv_path = Path(data_dir) / f'den_graph_data_{file_ext}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)

    edges_list = [ast.literal_eval(e) for e in df['EDGES']]

    graph_infos = []
    for edges in edges_list:
        nodes = sorted(set(u for e in edges for u in e))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        edge_indices = []
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        graph_infos.append({
            'num_nodes': len(nodes),
            'edge_list': edge_indices,
            'node_labels': nodes
        })

    return graph_infos




def check_feature_consistency(file_ext, data_dir=data_dir):
    """Check that all features have consistent shapes."""
    features_dir = Path(data_dir) / f'features_loop_{file_ext}'
    
    if not features_dir.exists():
        print(f"No features found for loop {file_ext}")
        return
    
    feature_files = list(features_dir.glob('*.npy'))
    
    print(f"Checking features for loop {file_ext}:")
    shapes = {}
    for f in feature_files:
        arr = np.load(f)
        shapes[f.stem] = arr.shape
        print(f"  {f.stem}: {arr.shape}")
    
    # Check consistency
    if len(set(shapes.values())) > 1:
        print("WARNING: Inconsistent feature shapes!")
    else:
        print("All features have consistent shapes.")


if __name__ == "__main__":
    # Example usage
    print("Testing feature loading...")
    import argparse
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--file_ext', type=str, help='Loop order (overrides config)')
    parser.add_argument('--feature', type=str, help='Specific feature to compute')
    parser.add_argument('--chunk-size', type=int, help='Chunk size (overrides config)')
    parser.add_argument('--n-jobs', type=int, help='Number of jobs (overrides config)')
    args = parser.parse_args()

    # Default config
    config = {
        'data': {
            'file_ext': 7,
            'base_dir': '../Graph_Edge_Data',        },
        'features': {
            'chunk_size': 100,
            'n_jobs': 1,
            'compute_all': False,
            'features_to_compute': []
        }
    }

    # Load YAML config if provided
    if args.config:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
        for section in user_config:
            if section in config:
                config[section].update(user_config[section])
            else:
                config[section] = user_config[section]

    # Set chunk_size, n_jobs, base_dir and file_ext
    file_ext = args.file_ext if args.file_ext else config['data']['file_ext']
    chunk_size = args.chunk_size if args.chunk_size else config['features']['chunk_size']
    n_jobs = args.n_jobs if args.n_jobs else config['features']['n_jobs']
    base_dir = Path(config['data'].get('base_dir', '../Graph_Edge_Data'))

    # --- Auto-tune fallback if not passed ---
    import psutil
    if not args.n_jobs and config['features']['n_jobs'] <= 1:
        n_cpus = psutil.cpu_count(logical=True)
        config['features']['n_jobs'] = int(0.75 * n_cpus)
    if not args.chunk_size or config['features']['chunk_size'] <= 100:
        mem_gb = psutil.virtual_memory().total / 1e9
        if mem_gb < 128:
            config['features']['chunk_size'] = 2000
        elif mem_gb < 256:
            config['features']['chunk_size'] = 5000
        elif mem_gb < 512:
            config['features']['chunk_size'] = 10000
        elif mem_gb < 768:
            config['features']['chunk_size'] = 20000
        else:
            config['features']['chunk_size'] = 30000

    
    # Check what features are available 
    available = get_available_features(file_ext, data_dir=base_dir)
    print(f"\nAvailable features for loop {file_ext}: {available}")
    
    if available:
        # Load some features
        features, labels = load_saved_features(file_ext, ['degree'])
        print(f"\nLoaded {len(labels)} graphs")
        print(f"[INFO] Final n_jobs={config['features']['n_jobs']}, chunk_size={config['features']['chunk_size']}")
    
    # Check consistency
    check_feature_consistency(file_ext)
