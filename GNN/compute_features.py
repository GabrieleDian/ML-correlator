import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import isomorphism
import ast
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import eigh


def load_graph_edges(loop_order, base_dir=None, extra_train=False):
    """
    Load graphs for a loop_order. If extra_train=True, only load files of the form
    den_graph_data_{loop_order}to*.csv
    """
    if base_dir is None:
        base_dir = BASE_DIR if 'BASE_DIR' in globals() else Path('../Graph_Edge_Data')

    edges_list = []
    labels = []

    if extra_train:
        # Only extra graphs
        pattern = base_dir / f'den_graph_data_{loop_order}to*.csv'
        files = list(pattern.parent.glob(pattern.name))
        if not files:
            raise FileNotFoundError(f"No extra files found matching {pattern}")
        for f in files:
            df = pd.read_csv(f)
            edges_list.extend([ast.literal_eval(e) for e in df['EDGES']])
            labels.extend(df['COEFFICIENTS'].tolist())
    else:
        # Only standard graphs
        file_path = base_dir / f'den_graph_data_{loop_order}.csv'
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
    """
    Compute face count features for a batch of planar graphs.
    For each node, count how many distinct faces (including outer face) it belongs to.
    """
    features = []

    for edges in graphs_batch:
        # Build graph: support edges as (u,v) or (u,v,w)
        G = nx.Graph()
        for e in edges:
            if len(e) == 2:
                u, v = e
            elif len(e) == 3:
                u, v, _ = e
            else:
                raise ValueError(f"Unexpected edge format: {e}")
            G.add_edge(u, v)

        # Ensure nodes are labeled consistently
        nodes = sorted(G.nodes())
        face_count = {n: 0 for n in nodes}

        # Get planar embedding
        is_planar, embedding = nx.check_planarity(G)
        if not is_planar:
            raise ValueError("Graph is not planar!")

        # Enumerate faces
        faces = list(embedding.faces())

        # Count node appearances in faces
        for face in faces:
            for n in face:
                face_count[n] += 1

        # Append features in node order
        features.append([face_count[n] for n in nodes])

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
    'W5_indicator': compute_W5_features
}
def pad_features(features_list, max_nodes):
    """Pad features to have consistent size across all graphs."""
    padded = []
    for features in features_list:
        if len(features) < max_nodes:
            # Pad with zeros
            padded_features = features + [0] * (max_nodes - len(features))
        else:
            padded_features = features[:max_nodes]
        padded.append(padded_features)
    return np.array(padded)



def compute_and_save_feature(feature_name, loop_order, chunk_size=1000, n_jobs=4,
                             extra_train=False, base_dir=None):
    if base_dir is None:
        base_dir = Path('../Graph_Edge_Data')

    # Folder naming
    folder_name = f"features_loop_{loop_order}to" if extra_train else f"features_loop_{loop_order}"
    output_dir = base_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{feature_name}.npy'
    if output_file.exists():
        print(f"Feature {feature_name} already computed for {folder_name}. Skipping.")
        return

    print(f"Computing {feature_name} for {folder_name}...")

    # Load edges
    edges_list, _ = load_graph_edges(loop_order, base_dir, extra_train=extra_train)

    n_graphs = len(edges_list)
    max_nodes = max(len(set(u for e in edges for u in e)) for edges in edges_list)

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


    features_array = pad_features(all_features, max_nodes)
    np.save(output_file, features_array)
    print(f"Saved {feature_name} features to {output_file}")



def compute_all_features(loop_order, chunk_size=1000, n_jobs=4, extra_train=False, base_dir=None):
    folder_name = f"loop {loop_order}to" if extra_train else f"loop {loop_order}"
    print(f"Computing all features for {folder_name}...")

    for feature_name in FEATURE_FUNCTIONS.keys():
        compute_and_save_feature(
            feature_name,
            loop_order,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            extra_train=extra_train,
            base_dir=base_dir
        )

    print(f"All features computed for {folder_name}!")


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--loop', type=int, help='Loop order (overrides config)')
    parser.add_argument('--feature', type=str, help='Specific feature to compute')
    parser.add_argument('--chunk-size', type=int, help='Chunk size (overrides config)')
    parser.add_argument('--n-jobs', type=int, help='Number of jobs (overrides config)')
    args = parser.parse_args()

    # Default config
    config = {
        'data': {
            'train_loop_order': None,
            'test_loop_order': None,
            'base_dir': '../Graph_Edge_Data',
            'extra_train': False
        },
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

    # Determine loop_order_list
    if args.loop is not None:
        loop_order_list = [args.loop]
    elif config['data'].get('train_loop_order') is not None:
        loop_order_list = config['data']['train_loop_order']
        if isinstance(loop_order_list, int):
            loop_order_list = [loop_order_list]
    elif config['data'].get('test_loop_order') is not None:
        loop_order_list = config['data']['test_loop_order']
        if isinstance(loop_order_list, int):
            loop_order_list = [loop_order_list]
    else:
        loop_order_list = [1]  # fallback

    # Determine extra_train from config (strict boolean)
    extra_train = config['data'].get('extra_train', False)
    if not isinstance(extra_train, bool):
        raise ValueError("extra_train in config must be true or false (boolean)")

    # Set chunk_size, n_jobs, base_dir
    chunk_size = args.chunk_size if args.chunk_size else config['features']['chunk_size']
    n_jobs = args.n_jobs if args.n_jobs else config['features']['n_jobs']
    base_dir = Path(config['data'].get('base_dir', '../Graph_Edge_Data'))

    # Iterate over loops
    for loop in loop_order_list:
        if args.feature:
            # Compute a single feature
            compute_and_save_feature(
                args.feature,
                loop_order=loop,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
                extra_train=extra_train,
                base_dir=base_dir
            )
        else:
            # Compute multiple or all features
            if config['features']['compute_all']:
                compute_all_features(
                    loop_order=loop,
                    chunk_size=chunk_size,
                    n_jobs=n_jobs,
                    extra_train=extra_train,
                    base_dir=base_dir
                )
            else:
                for feature_name in config['features']['features_to_compute']:
                    compute_and_save_feature(
                        feature_name,
                        loop_order=loop,
                        chunk_size=chunk_size,
                        n_jobs=n_jobs,
                        extra_train=extra_train,
                        base_dir=base_dir
                    )
