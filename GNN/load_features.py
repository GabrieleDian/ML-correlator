"""
Simple functions to load pre-computed features.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import ast
from joblib import Parallel, delayed
import math
data_dir = '../Graph_Edge_Data'


import psutil

def autotune_resources():
    """
    Automatically determine (n_jobs, chunk_size) based on hardware capacity.

    Strategy:
    - Use more workers on large-memory, many-core machines (HPC nodes).
    - Increase chunk_size aggressively when RAM is abundant.
    - Constrain chunk_size on machines with <64 GB RAM to avoid OOM.
    - Ensure that chunk_size is always large enough to keep workers busy.
    """

    import psutil

    cpu = psutil.cpu_count(logical=True)
    mem_gb = psutil.virtual_memory().total / (1024**3)

    # -----------------------------
    # Base heuristics for n_jobs
    # -----------------------------
    if mem_gb < 16:
        # Laptop / low-end workstation
        n_jobs = max(1, cpu // 4)
    elif mem_gb < 64:
        # Mid-range workstation
        n_jobs = max(2, cpu // 2)
    elif mem_gb < 128:
        # Larger workstation or small compute node
        n_jobs = max(4, int(cpu * 0.75))
    else:
        # HPC node with plenty of RAM
        n_jobs = cpu

    # -----------------------------
    # Base heuristics for chunk_size
    # -----------------------------
    # In your pipeline, a graph’s edge list is small (typically ~100 bytes).
    # The real memory cost comes from Python overhead + processed dicts.
    #
    # Safe rules:
    # - 1,000 graphs ≈ 0.1–0.3 GB peak
    # - 5,000 graphs ≈ 0.4–0.7 GB peak
    # - 20,000 graphs ≈ 1.5–3.0 GB peak
    # - 50,000 graphs ≈ 4–7 GB peak
    #
    # Use these empirical values to avoid RAM exhaustion.
    #

    if mem_gb < 16:
        chunk_size = 2000      # conservative
    elif mem_gb < 32:
        chunk_size = 5000
    elif mem_gb < 64:
        chunk_size = 8000
    elif mem_gb < 128:
        chunk_size = 20000     # safe for large nodes
    elif mem_gb < 256:
        chunk_size = 70000     # HPC-class
    else:
        chunk_size = 200000     # extreme machines (≥256 GB)

    # --------------------------------------------
    # Additional safety margin:
    # Avoid chunk_size so large that it overwhelms
    # parallel workers or the Python GIL overhead.
    # --------------------------------------------
    chunk_size = int(min(chunk_size, 3 * cpu * 1000))  # safety cap

    print(f"[AUTO] mem_gb={mem_gb:.1f}, cpu={cpu} → n_jobs={n_jobs}, chunk_size={chunk_size}")
    return n_jobs, chunk_size



def load_saved_features(file_ext, feature_names, data_dir="../Graph_Edge_Data",
                        n_jobs=None, chunk_size=None):
    """
    Load pre-computed features for a given loop order, and load labels either
    from the CSV (if available) or from the NPZ (preferred for large datasets).
    """

    data_dir = Path(data_dir)

    # Auto compute n_jobs and chunk_size if not provided
    if n_jobs is None:
        n_jobs, chunk_size = autotune_resources()

    # ---------------------------
    # 1. Match NPZ file exactly
    # ---------------------------
    npz_path = Path(data_dir) / f"den_graph_data_{file_ext}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Expected NPZ file not found: {npz_path}\n"
            f"Ensure you have den_graph_data_{file_ext}.npz in {data_dir}"
        )


    # ---------------------------
    # 2. Load labels from NPZ
    # ---------------------------
    data = np.load(npz_path, allow_pickle=True)
    
   # Strict uniformity: labels must come from 'coefficients'
    if "coefficients" not in data:
        raise KeyError(
            f"'coefficients' array not found inside {npz_path}. "
            "Dataset must contain 'edges' and 'coefficients'."
        )

    labels = data["coefficients"].tolist()


    # ---------------------------
    # 3. Load feature arrays normally
    # ---------------------------
    features_dir = data_dir / f"features_loop_{file_ext}"

    def load_one(feature_name):
        feature_path = features_dir / f"{feature_name}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature {feature_name} not found in {features_dir}. "
                "Run compute_features.py first."
            )
        arr = np.load(feature_path, mmap_mode="r")
        return feature_name, arr

    print(f"[INFO] Parallel loading {len(feature_names)} features with n_jobs={n_jobs}")

    loaded = Parallel(n_jobs=n_jobs)(
        delayed(load_one)(fname) for fname in feature_names
    )
    features = dict(loaded)

    for name, arr in features.items():
        print(f"Loaded {name}: shape {arr.shape}")

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



def load_graph_structure(file_ext,
                         data_dir="../Graph_Edge_Data",
                         n_jobs=None,
                         chunk_size=None):
    """
    Memory-safe, parallel loader for large .npz graph datasets.

    Args:
        file_ext (str): loop order (e.g., '7', '8')
        data_dir (str or Path): base directory
        n_jobs (int): number of parallel workers
        chunk_size (int): how many graphs to process per batch in memory

    Returns:
        list[dict]: each dict has keys num_nodes, edge_list, node_labels
    """
    # Auto compute n_jobs and chunk_size if not provided
    if n_jobs is None:
        n_jobs, chunk_size = autotune_resources()
    npz_path = Path(data_dir) / f"den_graph_data_{file_ext}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    print(f"[INFO] Loading large .npz file {npz_path.name} "
          f"with n_jobs={n_jobs}, chunk_size={chunk_size}")

    data = np.load(npz_path, allow_pickle=True)
    if "edges" not in data and "edge_list" not in data:
        raise KeyError("No 'edges' or 'edge_list' found in .npz file.")
    edges_list = data.get("edges", data.get("edge_list"))

    n_graphs = len(edges_list)
    print(f"[INFO] Found {n_graphs:,} graphs in {npz_path.name}")

    def process_edges(edges):
        edge_indices = np.array(edges, dtype=int)
        rev = edge_indices[:, [1, 0]]
        edge_indices = np.vstack([edge_indices, rev])
        edge_indices -= edge_indices.min()   # ensure 0-based indexing
        nodes = np.unique(edge_indices)
        return {
            "num_nodes": len(nodes),
            "edge_list": edge_indices.tolist(),
            "node_labels": nodes.tolist(),
        }

    graph_infos = []
    n_chunks = math.ceil(n_graphs / chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_graphs)
        subset = edges_list[start:end]

        results = Parallel(n_jobs=n_jobs,prefer="processes")(
            delayed(process_edges)(edges) for edges in subset
        )
        graph_infos.extend(results)

        print(f"[INFO] Processed chunk {i+1}/{n_chunks} "
              f"({end-start} graphs)")

    print(f"[INFO] Finished loading {len(graph_infos):,} graphs.")
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

    
    # Check what features are available 
    available = get_available_features(file_ext, data_dir=base_dir)
    print(f"\nAvailable features for loop {file_ext}: {available}")
    
    if available:
        # Load some features
        features, labels = load_saved_features(file_ext, ['degree'])
        print(f"\nLoaded {len(labels)} graphs")
        print(f" Final n_jobs={config['features']['n_jobs']}, chunk_size={config['features']['chunk_size']}")
    
    # Check consistency
    check_feature_consistency(file_ext)
