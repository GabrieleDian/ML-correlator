import numpy as np
import pandas as pd
import networkx as nx
import ast
from pathlib import Path
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
# Optimize chunk_size and n_jobs based on system resources
import psutil

def autotune_resources(chunk_size_default=100, n_jobs_default=1):
    """
    Auto-adjust chunk_size and n_jobs based on node CPU and memory specs.
    Returns (chunk_size, n_jobs) optimized for current environment.
    """
    n_cpus = psutil.cpu_count(logical=True)
    mem_gb = psutil.virtual_memory().total / 1e9

    # Use ~75% of available CPUs
    n_jobs = max(1, int(0.75 * n_cpus))

    # Adaptive chunk_size scaling based on memory
    if mem_gb < 128:
        chunk_size = 70000
    elif mem_gb < 256:
        chunk_size = 120000
    elif mem_gb < 512:
        chunk_size = 250000
    elif mem_gb < 768:
        chunk_size = 500000
    else:
        chunk_size = 1000000  # ≥700 GB nodes

    # Fall back to defaults if machine is tiny
    chunk_size = max(chunk_size, chunk_size_default)
    n_jobs = max(n_jobs, n_jobs_default)

    print(f"[AUTOTUNE] Detected {n_cpus} CPUs, {mem_gb:.1f} GB RAM → "
          f"n_jobs={n_jobs}, chunk_size={chunk_size}")

    return chunk_size, n_jobs


def load_graph_edges(file_ext='7', base_dir=None):
    """
    Load graph edges and coefficients for denominator graphs.

    Preferred format:
        den_graph_data_{file_ext}.npz
        ├─ edges          → np.array(list of edge lists)
        └─ coefficients   → np.array of floats/ints

    Fallback format:
        den_graph_data_{file_ext}.csv
        ├─ EDGES column         (stringified list of tuples)
        └─ COEFFICIENTS column  (numeric)

    Returns:
        edges_list: list of lists of (u, v) tuples
        coefficients: list of floats or ints
    """
    base_dir = Path(base_dir or '../Graph_Edge_Data')
    npz_path = base_dir / f'den_graph_data_{file_ext}.npz'
    csv_path = base_dir / f'den_graph_data_{file_ext}.csv'

    # --- Preferred: load from .npz
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        edges_list = data['edges'].tolist()
        coefficients = data['coefficients'].tolist()
        print(f"📦 Loaded {len(edges_list)} graphs from {npz_path.name}")
        return edges_list, coefficients

    # --- Fallback: load from .csv
    elif csv_path.exists():
        print(f"⚠️ Using CSV (no .npz found) for loop {file_ext}")
        df = pd.read_csv(csv_path)

        if 'EDGES' not in df.columns or 'COEFFICIENTS' not in df.columns:
            raise ValueError(f"CSV {csv_path} must contain 'EDGES' and 'COEFFICIENTS' columns")

        edges_list = [ast.literal_eval(e) for e in df['EDGES']]
        coefficients = df['COEFFICIENTS'].tolist()
        return edges_list, coefficients

    else:
        raise FileNotFoundError(f"No graph data found for loop {file_ext} in {base_dir}")




def edges_to_networkx(edges):
    """Convert edge list to NetworkX graph."""
    # Get all unique nodes
    nodes = sorted(set(u for e in edges for u in e))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    
    # Add edges with node indices
    for u, v in edges:
        G.add_edge(node_to_idx[u], node_to_idx[v])
    
    return G, len(nodes)

# Number of eigenvectors to compute

def compute_extreme_eigenvectors(graphs_batch, k_low=3, k_high=3):
    """
    Compute the k_low smallest (non-zero) and k_high largest Laplacian eigenvectors for each graph.
    Returns a list of (n_nodes, k_low + k_high) arrays per graph.
    """
    feats = []
    for edges in graphs_batch:
        G, n = edges_to_networkx(edges)
        L = nx.laplacian_matrix(G).astype(float)

        # Try efficient sparse solver; fallback to dense if fails
        try:
            # Smallest non-zero and largest eigenvectors
            vals_low, vecs_low = eigsh(L, k=min(k_low + 1, n-1), which='SM')  # Smallest magnitude
            vals_high, vecs_high = eigsh(L, k=min(k_high, n-1), which='LM')   # Largest magnitude
        except Exception:
            L = L.toarray()
            vals, vecs = eigh(L)
            # Sort ascending (low first)
            idx = np.argsort(vals)
            vals, vecs = vals[idx], vecs[:, idx]
            # Separate small and large
            vals_low, vecs_low = vals, vecs
            vals_high, vecs_high = vals, vecs

        # Filter out the zero eigenvalue from the low end
        nz = np.where(vals_low > 1e-10)[0][:k_low]
        low_vecs = vecs_low[:, nz] if len(nz) > 0 else np.zeros((n, k_low))

        # Take top k_high largest
        high_vecs = vecs_high[:, -k_high:] if vecs_high.shape[1] >= k_high else np.zeros((n, k_high))

        # Combine and round for numerical stability
        combined = np.round(np.real(np.hstack([low_vecs, high_vecs])), 8)
        feats.append(combined)

    return feats
# Create eigenvector functions for different k values
def create_eigen_feature_function(i, kind='low', k_low=3, k_high=3):
    """
    Create function returning a specific eigenvector column (low_i or high_i).
    """
    def f(batch):
        all_feats = compute_extreme_eigenvectors(batch, k_low=k_low, k_high=k_high)
        if kind == 'low':
            return [feat[:, i-1].tolist() for feat in all_feats]
        else:
            offset = k_low  # offset into the combined array
            return [feat[:, offset + i - 1].tolist() for feat in all_feats]
    return f

low_eigenvector_functions = {f'low_eigen_{i}': create_eigen_feature_function(i, 'low') for i in range(1, 4)}
eigenvector_functions = {f'eigen_{i}': create_eigen_feature_function(i, 'high') for i in range(1, 4)}



# Compute degree features for a batch of graphs
def compute_degree_features(graphs_batch):
    """Compute degree for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        degrees = [G.degree(i) for i in range(n_nodes)]
        features.append(degrees)
    
    return features
# Compute adjacency matrix columns for a batch of graphs
def adjacency_column_features(graphs_batch):
    """Compute adjacency matrix columns for a batch of graphs."""
    features = []
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        adj_matrix = nx.adjacency_matrix(G, nodelist=range(n_nodes)).toarray()
        # Each node gets its corresponding column
        adj_columns = [adj_matrix[:, i] for i in range(n_nodes)]
        features.append(adj_columns)
    return features

# Compute identity matrix columns for a batch of graphs
def identity_column_features(graphs_batch):
    """Compute identity matrix columns for a batch of graphs."""
    features = []
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        # Create identity matrix columns
        identity_matrix = np.eye(n_nodes)
        identity_columns = [identity_matrix[:, i] for i in range(n_nodes)]
        features.append(identity_columns)
    return features

# Compute betweenness, clustering, closeness, pagerank, and face count features
def compute_betweenness_features(graphs_batch):
    """Compute betweenness centrality for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        betweenness = nx.betweenness_centrality(G)
        betweenness_list = [betweenness[i] for i in range(n_nodes)]
        features.append(betweenness_list)
    
    return features


def compute_clustering_features(graphs_batch):
    """Compute clustering coefficient for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        clustering = nx.clustering(G)
        clustering_list = [clustering[i] for i in range(n_nodes)]
        features.append(clustering_list)

    return features


def compute_closeness_features(graphs_batch):
    """Compute closeness centrality for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        closeness = nx.closeness_centrality(G)
        closeness_list = [closeness[i] for i in range(n_nodes)]
        features.append(closeness_list)
    
    return features


def compute_pagerank_features(graphs_batch):
    """Compute PageRank for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        pagerank = nx.pagerank(G)
        pagerank_list = [pagerank[i] for i in range(n_nodes)]
        features.append(pagerank_list)
    
    return features




def compute_graphlet_features(graphs_batch, k=4, sizev=1, sizee=2, connect=True):
    """
    Compute local graphlet frequencies for the single-edge (k=2) graphlet using GEOMINE.
    - Assumes one edge type (edge present=1, absent=0) → sizee=2.
    - Returns, for each graph, a list of per-node counts (participation in single-edge graphlets).
    - If GEOMINE is unavailable, the function prints an error and exits.

    Args:
        graphs_batch: list of graphs; each graph is a list of undirected edges [(u, v), ...]
        k (int): graphlet number of nodes
        sizev (int): number of vertex colors (1 if unused)
        sizee (int): number of edge symbols INCLUDING 0 (no-edge). For one edge type use 2.
        connect (bool): count only connected graphlets (True)

    Returns:
        List[List[int]]: per-graph list of per-node local counts
    """
    try:
        import GEOMINE
    except Exception as e:
        import sys
        print("[ERROR] GEOMINE is required for 'local_single_edge_graphlet' feature. "
              "Install it via: pip install GEOMINE", file=sys.stderr)
        print(f"Cause: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)

    features = []
    for edges in graphs_batch:
        G, n = edges_to_networkx(edges)
        adj_matrix = nx.adjacency_matrix(G, nodelist=range(n)).toarray()
       # Per-node local counts via GEOMINE.Count
        per_node = []
        for ref in range(n):
            vec = np.asarray(GEOMINE.Count(adj_matrix, ref, k, sizev, sizee, connect), dtype=float)
            # For k=2 & one edge type, vec has length 1; summing is robust.
            per_node.append(vec)
        features.append(per_node)

    return features
# Wrapper functions for specific graphlet sizes
def graphlet_3(graphs_batch):
    return compute_graphlet_features(graphs_batch, k=3)

def graphlet_4(graphs_batch):
    return compute_graphlet_features(graphs_batch, k=4)

def graphlet_5(graphs_batch):
    return compute_graphlet_features(graphs_batch, k=5)


# Create a seperate dictionary for the eigenvector features
#eigenvector_dict = {f'eigen_{i}': compute_top_k_eigenvector(k=k, i=i) for i in range(k)}

# Dictionary mapping feature names to their computation functions
FEATURE_FUNCTIONS = {
    **eigenvector_functions,  # Add eigenvector functions 
    **low_eigenvector_functions,
    'identity_columns': identity_column_features,
    'adjacency_columns': adjacency_column_features,
    'degree': compute_degree_features,
    'betweenness': compute_betweenness_features,
    'clustering': compute_clustering_features,
    'closeness': compute_closeness_features,
    'pagerank': compute_pagerank_features,
    'graphlet_4': graphlet_4,
    'graphlet_5': graphlet_5
}

import numpy as np

# Combine features of different dimensions
def pad_features(features_list, max_nodes):
    padded = []
    for features in features_list:
        features = np.array(features)
        n_nodes = features.shape[0]
        if n_nodes < max_nodes:
            pad_shape = (max_nodes - n_nodes,) + features.shape[1:]
            features_padded = np.vstack([features, np.zeros(pad_shape)])
        else:
            features_padded = features[:max_nodes]
        padded.append(features_padded)
    return np.array(padded)



def compute_or_load_features(file_ext, selected_features, base_dir, chunk_size, n_jobs):
    """
    Compute or bundle selected node features for denominator graphs.

    Behavior:
    - If a feature .npy is missing, compute it.
    - If .npz already exists, load and preserve its stored features.
    - Add any newly computed or existing .npy features into the .npz.
    - The .npz file lives inside features_loop_{file_ext}/ and grows cumulatively.
    """
    from joblib import Parallel, delayed

    base_dir = Path(base_dir)
    feat_dir = base_dir / f'features_loop_{file_ext}'
    feat_dir.mkdir(exist_ok=True)
    npz_path = feat_dir / f'features_loop_{file_ext}.npz'

    # ----------------------------------------------------------
    # Load existing .npz (if any)
    # ----------------------------------------------------------
    existing_npz = {}
    if npz_path.exists():
        print(f"📂 Found existing {npz_path.name}, loading current features...")
        with np.load(npz_path, allow_pickle=True) as npz_data:
            for key in npz_data.files:
                existing_npz[key] = npz_data[key]
        print(f"🧩 Existing features in archive: {list(existing_npz.keys())}")

    # ----------------------------------------------------------
    # Identify which features need to be computed
    # ----------------------------------------------------------
    already_in_npz = set(existing_npz.keys())
    existing_npys = {f.stem for f in feat_dir.glob("*.npy")}
    missing_features = [
        f for f in selected_features
        if f not in already_in_npz and f not in existing_npys
    ]

    if not missing_features:
        print(f"✅ All requested features already exist (in .npz or as .npy).")
    else:
        print(f"🧮 Missing features for loop {file_ext}: {missing_features}")
        edges_list, _ = load_graph_edges(file_ext, base_dir)
        n_graphs = len(edges_list)
        max_nodes = max(len(set(u for e in edges for u in e)) for edges in edges_list)

        for feat in missing_features:
            if feat not in FEATURE_FUNCTIONS:
                print(f"⚠️ Unknown feature {feat}, skipping.")
                continue

            func = FEATURE_FUNCTIONS[feat]
            all_feat = []
            for i in tqdm(range(0, n_graphs, chunk_size), desc=f"Computing {feat}"):
                chunk = edges_list[i:i+chunk_size]
                if n_jobs > 1:
                    subchunks = [chunk[j:j+100] for j in range(0, len(chunk), 100)]
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(func)(p) for p in subchunks
                    )
                    chunk_out = [f for sub in results for f in sub]
                else:
                    chunk_out = func(chunk)
                all_feat.extend(chunk_out)

            arr = pad_features(all_feat, max_nodes)
            np.save(feat_dir / f"{feat}.npy", arr)
            existing_npz[feat] = arr
            print(f"✅ Computed and saved {feat}.npy")

    # ----------------------------------------------------------
    # Add any requested features from existing .npy files
    # ----------------------------------------------------------
    for feat in selected_features:
        if feat in existing_npz:
            continue  # already loaded or computed
        path = feat_dir / f"{feat}.npy"
        if path.exists():
            existing_npz[feat] = np.load(path)
            print(f"📎 Added existing {feat}.npy to archive.")
        else:
            print(f"⚠️ Feature {feat} not found anywhere — skipping.")

    # ----------------------------------------------------------
    # Save the enriched archive
    # ----------------------------------------------------------
    np.savez_compressed(npz_path, **existing_npz)
    print(f"💾 Updated {npz_path.name} with {len(existing_npz)} total features.")




def compute_all_features(file_ext='7', base_dir='../Graph_Edge_Data', chunk_size=100, n_jobs=1):
    """
    Compute and bundle *all* known features for a given loop order.
    If they already exist as .npy, reuses them and saves everything in one .npz.
    """
    print(f"[ALL FEATURES] Computing all features for loop {file_ext}")
    all_feats = list(FEATURE_FUNCTIONS.keys())
    compute_or_load_features(
        file_ext=file_ext,
        selected_features=all_feats,
        base_dir=base_dir,
        chunk_size=chunk_size,
        n_jobs=n_jobs
    )



if __name__ == "__main__":
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
    # --- Auto-tune if user didn't override from command line ---
    if args.chunk_size or args.n_jobs:
        # Respect explicit user settings
        chunk_size = args.chunk_size or config['features']['chunk_size']
        n_jobs = args.n_jobs or config['features']['n_jobs']
    else:
        # Auto-detect based on hardware
        chunk_size, n_jobs = autotune_resources(
            chunk_size_default=config['features']['chunk_size'],
            n_jobs_default=config['features']['n_jobs']
        )

    base_dir = Path(config['data'].get('base_dir', '../Graph_Edge_Data'))

    
            # Compute multiple or all features
    if config['features']['compute_all']:
        compute_all_features(
            file_ext=file_ext,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            base_dir=base_dir
        )
    else:
        selected_features = config['features'].get('features_to_compute', [])
        if not selected_features:
            print("⚠️ No features_to_compute listed in config — computing all available ones.")
            selected_features = list(FEATURE_FUNCTIONS.keys())

        compute_or_load_features(
            file_ext=file_ext,
            selected_features=selected_features,
            base_dir=base_dir,
            chunk_size=chunk_size,
            n_jobs=n_jobs
        )
