"""
Simple functions to load pre-computed features.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import ast
from fractions import Fraction

data_dir = '../Graph_Edge_Data'
# Load features for multi-edge graphs
def load_saved_features(file_ext, feature_names, data_dir='../Graph_Edge_Data'):
    """
    Load pre-computed features (supports both bundled .npz and individual .npy files).
    Returns: features dict, labels list
    """
    from fractions import Fraction
    import numpy as np
    import pandas as pd
    from pathlib import Path

    features_dir = Path(data_dir) / f"f_features_loop_{file_ext}"
    npz_bundle = features_dir.with_suffix(".npz")
    csv_path = Path(data_dir) / f"graph_data_{file_ext}.csv"

    # Load labels
    df = pd.read_csv(csv_path)
    true_labels = df["COEFFICIENTS"].tolist()
    labels = [1 if Fraction(c) != 0 else 0 for c in true_labels]

    features = {}

    if npz_bundle.exists():
        print(f"✅ Loading bundled features from {npz_bundle.name}")
        bundle = np.load(npz_bundle, allow_pickle=True)
        for name in feature_names:
            if name in bundle.files:
                features[name] = bundle[name]
            else:
                print(f"⚠️ Feature {name} not found in bundle.")
    else:
        print(f"⚠️ No bundle found; loading individual .npy files (slower).")
        for name in feature_names:
            fpath = features_dir / f"{name}.npy"
            if not fpath.exists():
                raise FileNotFoundError(f"Feature {name} not found in {features_dir}")
            features[name] = np.load(fpath)
            print(f"Loaded {name}: shape {features[name].shape}")

    return features, labels




def get_available_features(loop_order, data_dir='../Graph_Edge_Data'):
    """
    Return all available features (from .npz or .npy).
    """
    from pathlib import Path
    import numpy as np

    features_dir = Path(data_dir) / f"f_features_loop_{loop_order}"
    npz_path = features_dir.with_suffix(".npz")

    if npz_path.exists():
        bundle = np.load(npz_path, allow_pickle=True)
        available = sorted(bundle.files)
        print(f"✅ Found {len(available)} features in {npz_path.name}")
        return available

    elif features_dir.exists():
        feature_files = list(features_dir.glob("*.npy"))
        available = sorted([f.stem for f in feature_files])
        print(f"✅ Found {len(available)} .npy feature files in {features_dir.name}")
        return available

    else:
        print(f"⚠️ No features found for loop {loop_order}")
        return []



def load_graph_structure(file_ext, data_dir=None):
    """
    Load original multi-edge graph edges for creating edge_index for GNNs.
    Returns a list of dicts with:
        'num_nodes', 'edge_list', 'edge_types', 'node_labels'
    """
    base_dir = Path(data_dir or '../Graph_Edge_Data')
    npz_path = base_dir / f'graph_edges_{file_ext}.npz'
    csv_path = base_dir / f'graph_data_{file_ext}.csv'

    if npz_path.exists():
        print(f"✅ Loaded graph structure from {npz_path.name}")
        data = np.load(npz_path, allow_pickle=True)
        denom_edges_list = data['denom_edges']
        numer_edges_list = data['numer_edges']
    elif csv_path.exists():
        print(f"⚠️ No preprocessed file found for loop {file_ext}, using CSV (slow).")
        df = pd.read_csv(csv_path)
        denom_edges_list = [ast.literal_eval(e) for e in df['DEN_EDGES']]
        numer_edges_list = [ast.literal_eval(e) for e in df['NUM_EDGES']]
    else:
        raise FileNotFoundError(f"Neither {npz_path} nor {csv_path} found.")

    graph_infos = []
    for d_edges, n_edges in zip(denom_edges_list, numer_edges_list):
        d_edges = list(d_edges)
        n_edges = list(n_edges)
        nodes = sorted(set([u for u, v in d_edges + n_edges] + [v for u, v in d_edges + n_edges]))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        edge_indices = []
        edge_types = []
        edge_dict = {}

        # Add denominator edges (-1)
        for u, v in d_edges:
            key = tuple(sorted((u, v)))
            edge_dict[key] = -1

        # Add numerator edges (1, 2, 3, ...)
        for u, v in n_edges:
            key = tuple(sorted((u, v)))
            if key in edge_dict and edge_dict[key] == -1:
                raise ValueError(f"Conflict: both numerator and denominator edge between {key}")
            edge_dict[key] = edge_dict.get(key, 0) + 1

        for (u, v), etype in edge_dict.items():
            i, j = node_to_idx[u], node_to_idx[v]
            edge_indices += [[i, j], [j, i]]
            edge_types += [etype, etype]

        graph_infos.append({
            'num_nodes': len(nodes),
            'edge_list': edge_indices,
            'edge_types': edge_types,
            'node_labels': nodes,
        })

    return graph_infos





def check_feature_consistency(file_ext, data_dir=data_dir):
    """Check that all features have consistent shapes."""
    features_dir = Path(data_dir) / f'f_features_loop_{file_ext}'
    
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

    # Get the avalable features
    available = get_available_features(file_ext, data_dir=base_dir)
    print(f"\nAvailable features for loop {file_ext}: {available}")

    # Load selected features from config instead of hardcoded 'degree'
    selected_features = config.get('data', {}).get('selected_features', ['degree'])
    print(f"Loading features: {selected_features}")
    features, labels = load_saved_features(file_ext, selected_features)

    print(f"\nLoaded {len(labels)} graphs")
    print(f"First graph degree features: {features['degree'][0]}")

    # Check consistency
    check_feature_consistency(file_ext)
