import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import isomorphism
import ast
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import eigh


def load_graph_edges(file_ext='7', base_dir=None):
    """
    Load graph edges and labels from CSV file.
    """
    if base_dir is None:
        base_dir = Path('../Graph_Edge_Data')

    edges_list = []
    labels = []
    # Only standard graphs
    file_path = base_dir / f'den_graph_data_{file_ext}.csv'
    df = pd.read_csv(file_path)
    edges_list = [ast.literal_eval(e) for e in df['EDGES']]
    labels = df['COEFFICIENTS'].tolist()

    return edges_list, labels



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
k=3
# Compute the top k eigenvectors for the regular Laplacian of each graph
def compute_ith_eigenvector(graphs_batch, k=k,i=0):
    """Compute k eigenvectors with the largest eigenvalues for regular Laplacian."""
    # Initialize list to store k different features
    features = []

    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        
        # Get the regular Laplacian matrix
        laplacian_matrix = nx.laplacian_matrix(G).toarray()
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(laplacian_matrix)
        
        # Sort by eigenvalues in descending order (largest first)
        idx = np.argsort(np.real(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take the first k eigenvectors (corresponding to largest eigenvalues)
        top_k_eigenvectors = eigenvectors[:, :k]
        
        # Take real parts
        top_k_eigenvectors = np.round(np.real(top_k_eigenvectors),8)
        
        features.append(top_k_eigenvectors[:,i-1].tolist())
    return features

# Create a function that takes only the i-th output of the top_k_eigenvector function
def create_eigen_function(i):
    """Create a function that computes only the i-th eigenvector."""
    def eigen_function(graphs_batch):
        return compute_ith_eigenvector(graphs_batch, k=k, i=i)
    return eigen_function
# Create a dictionary of functions for each eigenvector
eigenvector_functions = {f'eigen_{i}': create_eigen_function(i) for i in range(1, k+1)}

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


import networkx as nx

def compute_face_count_features(graphs_batch):
    """Compute face count features for a batch of planar graphs."""
    features = []
    
    for edges in graphs_batch:
        # Build graph in the same way as your PageRank code
        G, n_nodes = edges_to_networkx(edges)

        # Check planarity
        is_planar, embedding = nx.check_planarity(G)
        if not is_planar:
            raise ValueError("Graph is not planar!")

        # Collect unique faces
        seen_faces = set()
        faces = []
        for u, v in embedding.edges():
            face = tuple(embedding.traverse_face(u, v))
            # Normalize face so duplicates are detected
            min_idx = face.index(min(face))
            normalized = face[min_idx:] + face[:min_idx]
            if normalized not in seen_faces:
                seen_faces.add(normalized)
                faces.append(face)

        # Count face participation per node
        face_count = {n: 0 for n in range(n_nodes)}
        for face in faces:
            for n in face:
                face_count[n] += 1

        # Store as list in node order
        face_count_list = [face_count[i] for i in range(n_nodes)]
        features.append(face_count_list)

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


    
def compute_W5_features(graphs_batch):
    W5 = nx.wheel_graph(5)

    results = []
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        has_W5 = int(isomorphism.GraphMatcher(G, W5).subgraph_is_isomorphic())
        results.append([has_W5] * n_nodes)
    return results

# Create a seperate dictionary for the eigenvector features
#eigenvector_dict = {f'eigen_{i}': compute_top_k_eigenvector(k=k, i=i) for i in range(k)}

# Dictionary mapping feature names to their computation functions
FEATURE_FUNCTIONS = {
    **eigenvector_functions,  # Add eigenvector functions 
    'identity_columns': identity_column_features,
    'adjacency_columns': adjacency_column_features,
    'degree': compute_degree_features,
    'betweenness': compute_betweenness_features,
    'clustering': compute_clustering_features,
    'closeness': compute_closeness_features,
    'pagerank': compute_pagerank_features,
    'face_count': compute_face_count_features,
    'graphlet_2': compute_graphlet_features,
    'W5_indicator': compute_W5_features
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



def compute_and_save_feature(feature_name, file_ext='7',chunk_size=1000, n_jobs=4,base_dir=None):
    if base_dir is None:
        base_dir = Path('../Graph_Edge_Data')

    # Folder naming
    folder_name = f"features_loop_{file_ext}" 
    output_dir = base_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{feature_name}.npy'
    if output_file.exists():
        print(f"Feature {feature_name} already computed for {folder_name}. Skipping.")
        return

    print(f"Computing {feature_name} for {folder_name}...")

    # Load edges
    edges_list, _ = load_graph_edges(file_ext, base_dir )

    n_graphs = len(edges_list)
    max_nodes = max(len(set(u for e in edges for u in e)) for edges in edges_list)
    print(f"Max nodes in graphs: {max_nodes}")
    if feature_name not in FEATURE_FUNCTIONS:
        print(f"Feature {feature_name} not recognized. Skipping.")
        return
    compute_func = FEATURE_FUNCTIONS[feature_name]
    all_features = []

    for i in tqdm(range(0, n_graphs, chunk_size)):
        chunk_edges = edges_list[i:i+chunk_size]
        if n_jobs > 1:
            sub_chunks = [chunk_edges[j:j+100] for j in range(0, len(chunk_edges), 100)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_func)(sub_chunk) for sub_chunk in sub_chunks
            )
            # Safely flatten results
            chunk_features = [f for sublist in results if sublist is not None for f in sublist]
        else:
            chunk_features = compute_func(chunk_edges)
            if chunk_features is None:
                chunk_features = []
        all_features.extend(chunk_features)

    features_array = pad_features(all_features, max_nodes)
    np.save(output_file, features_array)
    print(f"Saved {feature_name} features to {output_file}")



def compute_all_features(file_ext='7', chunk_size=1000, n_jobs=4, base_dir=None):
    folder_name = f"features_loop_{file_ext}"
    print(f"Computing all features for {folder_name}...")

    for feature_name in FEATURE_FUNCTIONS.keys():
        compute_and_save_feature(
            feature_name,
            file_ext=file_ext,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            base_dir=base_dir
        )

    print(f"All features computed for {folder_name}!")


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
    chunk_size = args.chunk_size if args.chunk_size else config['features']['chunk_size']
    n_jobs = args.n_jobs if args.n_jobs else config['features']['n_jobs']
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
        for feature_name in config['features']['features_to_compute']:
            compute_and_save_feature(
                feature_name,
                file_ext=file_ext,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
                base_dir=base_dir
                    )
