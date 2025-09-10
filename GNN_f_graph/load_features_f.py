"""
Simple functions to load pre-computed features.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import ast

data_dir = '../Graph_Edge_Data'
# Load features for multi-edge graphs
def load_saved_features(loop_order, feature_names, data_dir=None):
    """
    Load pre-computed features for a given loop order and multi-edge graphs.

    Args:
        loop_order: Loop order (e.g., 7, 8)
        feature_names: List of feature names to load
        data_dir: Base directory for data

    Returns:
        features: Dict mapping feature names to numpy arrays
        labels: List of graph labels
    """
    base_dir = data_dir if data_dir is not None else Path('../Graph_Edge_Data')
    features_dir = Path(base_dir) / f'f_features_loop_{loop_order}'
    # Load labels from the multi-edge CSV
    csv_path =  Path(base_dir) / f'graph_data_{loop_order}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file {csv_path} not found.")
    df = pd.read_csv(csv_path)
    labels = df['COEFFICIENTS'].tolist()


    # Debug
    print(f"Looking for features in: {features_dir.resolve()}")
    print("Existing files:", list(features_dir.glob("*.npy")))

    # Load requested features
    features = {}
    for feature_name in feature_names:
        feature_path = features_dir / f'{feature_name}.npy'
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature {feature_name} not found for loop {loop_order}. "
                f"Run compute_features.py first."
            )
        features[feature_name] = np.load(feature_path)
        print(f"Loaded {feature_name}: shape {features[feature_name].shape}")

    return features, labels



def get_available_features(loop_order, data_dir=data_dir):
    """Get list of available pre-computed features for a loop order."""
    features_dir = Path(data_dir) / f'f_features_loop_{loop_order}'
    
    if not features_dir.exists():
        return []
    
    # Check for .npy files
    feature_files = list(features_dir.glob('*.npy'))
    feature_names = [f.stem for f in feature_files]
    
    return sorted(feature_names)


def load_graph_structure(loop_order, data_dir=None):
    """
    Load original multi-edge graph edges for creating edge_index for GNNs.

    Returns a list of dicts with:
        'num_nodes': number of nodes
        'edge_list': list of [i,j] edge indices (bidirectional)
        'edge_types': list of edge types corresponding to each edge in edge_list
        'node_labels': original node labels
    """
    base_dir = data_dir if data_dir is not None else Path('../Graph_Edge_Data')
    csv_path = Path(base_dir) / f'graph_data_{loop_order}.csv'
    df = pd.read_csv(csv_path)

    # Parse edge lists
    denom_edges_list = [ast.literal_eval(e) for e in df['DEN_EDGES']]
    numer_edges_list = [ast.literal_eval(e) for e in df['NUM_EDGES']]

    graph_infos = []
    for d_edges, n_edges in zip(denom_edges_list, numer_edges_list):
        nodes = sorted(set([u for u,v in d_edges+n_edges] + [v for u,v in d_edges+n_edges]))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        edge_indices = []
        edge_types = []

        # Add denominator edges
        for u, v in d_edges:
            i, j = node_to_idx[u], node_to_idx[v]
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # bidirectional
            edge_types.extend([0, 0])

        # Add numerator edges
        for u, v in n_edges:
            i, j = node_to_idx[u], node_to_idx[v]
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_types.extend([1, 1])

        graph_infos.append({
            'num_nodes': len(nodes),
            'edge_list': edge_indices,
            'edge_types': edge_types,
            'node_labels': nodes
        })

    return graph_infos



def check_feature_consistency(loop_order, data_dir=data_dir):
    """Check that all features have consistent shapes."""
    features_dir = Path(data_dir) / f'f_features_loop_{loop_order}'
    
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
    loop = 7
    available = get_available_features(loop)
    print(f"\nAvailable features for loop {loop}: {available}")
    
    if available:
        # Load some features
        features, labels = load_saved_features(loop, ['degree'])
        print(f"\nLoaded {len(labels)} graphs")
        print(f"First graph degree features: {features['degree'][0]}")
    
    # Check consistency
    check_feature_consistency(loop)
