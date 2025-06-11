"""
Simple functions to load pre-computed features.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import ast

data_dir = '../Graph_Edge_Data'


def load_saved_features(loop_order, feature_names, data_dir=data_dir):
    """
    Load pre-computed features for a given loop order.
    
    Args:
        loop_order: Loop order (e.g., 7, 8)
        feature_names: List of feature names to load
        data_dir: Base directory for data
    
    Returns:
        features: Dict mapping feature names to numpy arrays
        labels: List of graph labels
    """
    features_dir = Path(data_dir) / f'features_loop_{loop_order}'
    
    # Load labels
    csv_path = Path(data_dir) / f'den_graph_data_{loop_order}.csv'
    df = pd.read_csv(csv_path)
    labels = df['COEFFICIENTS'].tolist()
    
    # Load requested features
    features = {}
    for feature_name in feature_names:
        feature_path = features_dir / f'{feature_name}.npy'
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature {feature_name} not found for loop {loop_order}. "
                                  f"Run compute_features.py first.")
        
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


def load_graph_structure(loop_order, data_dir=data_dir):
    """Load original graph edges for creating edge_index."""
    csv_path = Path(data_dir) / f'den_graph_data_{loop_order}.csv'
    df = pd.read_csv(csv_path)
    
    edges_list = [ast.literal_eval(e) for e in df['EDGES']]
    
    # Get node mapping for each graph
    graph_infos = []
    for edges in edges_list:
        nodes = sorted(set(u for e in edges for u in e))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Convert edges to indices
        edge_indices = []
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Bidirectional
        
        graph_infos.append({
            'num_nodes': len(nodes),
            'edge_list': edge_indices,
            'node_labels': nodes
        })
    
    return graph_infos


def check_feature_consistency(loop_order, data_dir=data_dir):
    """Check that all features have consistent shapes."""
    features_dir = Path(data_dir) / f'features_loop_{loop_order}'
    
    if not features_dir.exists():
        print(f"No features found for loop {loop_order}")
        return
    
    feature_files = list(features_dir.glob('*.npy'))
    
    print(f"Checking features for loop {loop_order}:")
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
    
    # Check what features are available
    loop = 8
    available = get_available_features(loop)
    print(f"\nAvailable features for loop {loop}: {available}")
    
    if available:
        # Load some features
        features, labels = load_saved_features(loop, ['degree'])
        print(f"\nLoaded {len(labels)} graphs")
        print(f"First graph degree features: {features['degree'][0]}")
    
    # Check consistency
    check_feature_consistency(loop)
