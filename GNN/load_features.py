"""
Utility functions to load pre-computed graph features
for denominator graphs (den_graph_data_X).
Now supports unified .npz archives that store all features together.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import ast

# Default base path
DATA_DIR = '../Graph_Edge_Data'


# ===============================================================
#  Load features (with .npz priority)
# ===============================================================
def load_saved_features(file_ext, feature_names=None, data_dir=DATA_DIR):
    """
    Load pre-computed features for a given loop order.

    Priority:
        1. features_loop_{file_ext}/features_loop_{file_ext}.npz
        2. Individual .npy files in the same folder.

    Args:
        file_ext (str or int): loop order (e.g. '7', 8)
        feature_names (list[str] or None): specific features to load.
                                           If None, load all available.
        data_dir (str): base directory (default ../Graph_Edge_Data)

    Returns:
        dict[str, np.ndarray]: {feature_name: feature_array}
    """
    base_dir = Path(data_dir)
    feat_dir = base_dir / f'features_loop_{file_ext}'
    npz_path = feat_dir / f'features_loop_{file_ext}.npz'

    features = {}

    # --- Case 1: unified .npz file exists
    if npz_path.exists():
        print(f"📦 Loading from {npz_path.name}")
        with np.load(npz_path, allow_pickle=True) as npz_data:
            available = list(npz_data.files)

            # If no specific list is given → load all
            if feature_names is None:
                feature_names = available

            for feat in feature_names:
                if feat in npz_data:
                    features[feat] = npz_data[feat]
                else:
                    print(f"⚠️ Feature {feat} not found in archive.")

        return features

    # --- Case 2: fallback to individual .npy files
    print(f"ℹ️ No .npz found for loop {file_ext}, loading individual .npy files.")
    if not feat_dir.exists():
        raise FileNotFoundError(f"No features folder found: {feat_dir}")

    npy_files = list(feat_dir.glob("*.npy"))
    available = [f.stem for f in npy_files]

    if feature_names is None:
        feature_names = available

    for feat in feature_names:
        path = feat_dir / f"{feat}.npy"
        if path.exists():
            features[feat] = np.load(path)
        else:
            print(f"⚠️ Feature {feat}.npy not found in {feat_dir}")

    return features


# ===============================================================
#  Helper: list available features
# ===============================================================
def get_available_features(file_ext, data_dir=DATA_DIR):
    """Return all available feature names (from .npz or .npy files)."""
    base_dir = Path(data_dir)
    feat_dir = base_dir / f'features_loop_{file_ext}'
    npz_path = feat_dir / f'features_loop_{file_ext}.npz'

    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as npz_data:
            return sorted(npz_data.files)

    if not feat_dir.exists():
        return []

    return sorted([f.stem for f in feat_dir.glob("*.npy")])


# ===============================================================
#  Load graph structure (unchanged)
# ===============================================================
def load_graph_structure(file_ext, data_dir=DATA_DIR):
    """
    Load edge information for denominator graphs.
    Prefers .npz format, falls back to .csv if missing.
    """
    base_dir = Path(data_dir)
    npz_path = base_dir / f'den_graph_data_{file_ext}.npz'
    csv_path = base_dir / f'den_graph_data_{file_ext}.csv'

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        edges_list = data['edges'].tolist()
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        edges_list = [ast.literal_eval(e) for e in df['EDGES']]
    else:
        raise FileNotFoundError(f"No graph data found for loop {file_ext}")

    graph_infos = []
    for edges in edges_list:
        nodes = sorted(set(u for e in edges for u in e))
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        edge_idx = []
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            edge_idx.append([i, j])
            edge_idx.append([j, i])
        graph_infos.append({'num_nodes': len(nodes), 'edge_list': edge_idx})
    return graph_infos



# ===============================================================
#  CLI test
# ===============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_ext", type=str, default="7")
    parser.add_argument("--feature_names", nargs="*", default=None)
    args = parser.parse_args()

    feats = load_saved_features(args.file_ext, args.feature_names)
    print(f"\n✅ Loaded features: {list(feats.keys())}")
    for k, v in feats.items():
        print(f"{k}: shape {v.shape}")
