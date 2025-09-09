import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import isomorphism
import ast
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import eigh


def load_graph_edges(loop_order, data_path=None):
    """Load graph edges with two types (denominator and numerator) from CSV file."""
    if data_path is None:
        base = BASE_DIR if 'BASE_DIR' in globals() else Path('../Graph_Edge_Data')
        data_path = base / f'graph_data_{loop_order}.csv'
    
    df = pd.read_csv(data_path)
    
    # Parse edge lists
    denom_edges = [ast.literal_eval(e) for e in df['DEN_EDGES']]
    numer_edges = [ast.literal_eval(e) for e in df['NUM_EDGES']]
    
    labels = df['COEFFICIENTS'].tolist()
    
    # Combine with edge type info: 0 = denominator, 1 = numerator
    edge_lists = []
    for d_edges, n_edges in zip(denom_edges, numer_edges):
        edges = []
        for e in d_edges:
            edges.append((e[0], e[1], 0))  # type 0
        for e in n_edges:
            edges.append((e[0], e[1], 1))  # type 1
        edge_lists.append(edges)
    
    return edge_lists, labels


def edges_to_networkx(edges):
    """Convert edge list with types to NetworkX graph."""
    # Get all unique nodes
    nodes = sorted(set([u for u, v, _ in edges] + [v for u, v, _ in edges]))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    
    # Add edges with type attribute
    for u, v, etype in edges:
        G.add_edge(node_to_idx[u], node_to_idx[v], edge_type=etype)
    
    return G, len(nodes)

import numpy as np
import networkx as nx
from scipy.linalg import eigh  # dense eigendecomposition

# Number of eigenvectors to compute
k = 3

def compute_ith_eigenvector(graphs_batch, k=k, i=0):
    """Compute top-k eigenvectors (largest eigenvalues) of the difference Laplacian."""
    features = []

    for edges in graphs_batch:
        # Collect nodes
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)

        # Build adjacency matrices
        A_denom = np.zeros((n, n))
        A_num   = np.zeros((n, n))
        for u, v, etype in edges:
            i_u, i_v = node_to_idx[u], node_to_idx[v]
            if etype == 0:   # denominator
                A_denom[i_u, i_v] = A_denom[i_v, i_u] = 1
            elif etype == 1: # numerator
                A_num[i_u, i_v] = A_num[i_v, i_u] = 1

        # Laplacians
        L_denom = np.diag(A_denom.sum(axis=1)) - A_denom
        L_num   = np.diag(A_num.sum(axis=1))   - A_num
        L_diff  = L_denom - L_num

        # Eigen-decomposition
        eigenvalues, eigenvectors = eigh(L_diff)

        # Sort eigenvalues in descending order
        idx = np.argsort(np.real(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take the top-k eigenvectors
        top_k_eigenvectors = eigenvectors[:, :k]

        # Round to avoid floating precision issues
        top_k_eigenvectors = np.round(np.real(top_k_eigenvectors), 8)

        # Take only the i-th one (1-based index, consistent with your code)
        features.append(top_k_eigenvectors[:, i-1].tolist())

    return features

def create_eigen_function(i):
    """Create a function that computes only the i-th eigenvector of L_denom - L_num."""
    def eigen_function(graphs_batch):
        return compute_ith_eigenvector(graphs_batch, k=k, i=i)
    return eigen_function

# Create a dictionary of functions for each eigenvector
eigenvector_functions = {f'eigen_{i}': create_eigen_function(i) for i in range(1, k+1)}


def compute_degree_features(graphs_batch):
    """Compute degree features based only on denominator edges (etype=0)."""
    features = []

    for edges in graphs_batch:
        # Collect nodes
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)

        # Initialize degree counts
        degrees = [0] * n

        # Count only denominator edges
        for u, v, etype in edges:
            if etype == 0:  # denominator edge
                i, j = node_to_idx[u], node_to_idx[v]
                degrees[i] += 1
                degrees[j] += 1

        features.append(degrees)

    return features
# Difference of adjacency matrices as features
def adjacency_column_features(graphs_batch):
    """Compute adjacency difference matrix columns (A_denom - A_num) for a batch of graphs."""
    features = []
    for edges in graphs_batch:
        # Collect nodes
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)

        # Build separate adjacency matrices
        A_denom = np.zeros((n, n))
        A_num   = np.zeros((n, n))
        for u, v, etype in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            if etype == 0:   # denominator
                A_denom[i, j] = A_denom[j, i] = 1
            elif etype == 1: # numerator
                A_num[i, j] = A_num[j, i] = 1

        # Difference adjacency
        A_diff = A_denom - A_num

        # Each node gets its corresponding column
        adj_columns = [A_diff[:, i].tolist() for i in range(n)]
        features.append(adj_columns)

    return features


# Betweeness centrality based on denominator edges only
def compute_betweenness_features(graphs_batch):
    """Compute betweenness centrality using only denominator edges (etype=0)."""
    features = []
    
    for edges in graphs_batch:
        # Collect nodes
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)

        # Build graph with denominator edges only
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, etype in edges:
            if etype == 0:  # denominator
                i, j = node_to_idx[u], node_to_idx[v]
                G.add_edge(i, j)

        # Compute betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        betweenness_list = [betweenness[i] for i in range(n)]
        features.append(betweenness_list)
    
    return features

# Clustering coefficient based on denominator edges only
def compute_closeness_features(graphs_batch):
    """Compute closeness centrality for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        closeness = nx.closeness_centrality(G)
        closeness_list = [closeness[i] for i in range(n_nodes)]
        features.append(closeness_list)
    
    return features


# PageRank based on denominator edges only
def compute_pagerank_features(graphs_batch):
    """Compute PageRank using only denominator edges (etype=0)."""
    features = []
    
    for edges in graphs_batch:
        # Collect nodes
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)

        # Build graph with denominator edges only
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, etype in edges:
            if etype == 0:  # denominator edge
                i, j = node_to_idx[u], node_to_idx[v]
                G.add_edge(i, j)

        # Compute PageRank
        pagerank = nx.pagerank(G)
        pagerank_list = [pagerank[i] for i in range(n)]
        features.append(pagerank_list)
        return features

# Face count feature based on planar embedding
def compute_face_count_features(graphs_batch):
    """Compute number of faces each node belongs to (planar graph feature),
    considering only denominator edges (etype=0)."""
    features = []
    
    for edges in graphs_batch:
        # Collect nodes
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)

        # Build graph with denominator edges only
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, etype in edges:
            if etype == 0:  # denominator edge
                i, j = node_to_idx[u], node_to_idx[v]
                G.add_edge(i, j)

        # Try planar embedding
        is_planar, embedding = nx.check_planarity(G)
        if not is_planar:
            # fallback: degree proxy
            face_counts = [G.degree(i) for i in range(n)]
        else:
            face_counts = [0] * n
            visited_edges = set()
            
            for node in embedding.nodes():
                for neighbor in embedding.neighbors(node):
                    edge = (node, neighbor) if node < neighbor else (neighbor, node)
                    if edge not in visited_edges:
                        face = list(embedding.traverse_face(node, neighbor))
                        for v in face:
                            if v < n:  # safety check
                                face_counts[v] += 1
                        # mark edges as visited
                        for i in range(len(face)):
                            u, v = face[i], face[(i + 1) % len(face)]
                            visited_edges.add((u, v) if u < v else (v, u))
        
        features.append(face_counts)
    
    return features
# W5 subgraph indicator feature with further check on numerator edges
import networkx as nx
from networkx.algorithms import isomorphism

def compute_W5_features(graphs_batch):
    W5 = nx.wheel_graph(5)
    results = []

    for edges in graphs_batch:
        # Build denominator-only graph
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)
        G_den = nx.Graph()
        G_den.add_nodes_from(range(n))
        for u, v, etype in edges:
            if etype == 0:
                G_den.add_edge(node_to_idx[u], node_to_idx[v])

        # Collect numerator edges
        num_edges = set()
        for u, v, etype in edges:
            if etype == 1:
                i, j = node_to_idx[u], node_to_idx[v]
                num_edges.add((min(i,j), max(i,j)))

        feature_value = 0
        matcher = isomorphism.GraphMatcher(G_den, W5)

        for mapping in matcher.subgraph_isomorphisms_iter():
            # Check that all W5 cycle nodes are present in mapping
            cycle_nodes = [0,1,2,3]
            if not all(n in mapping for n in cycle_nodes):
                continue

            mapped = [mapping[n] for n in cycle_nodes]

            # Opposite pairs
            opp_pairs = [(mapped[0], mapped[2]), (mapped[1], mapped[3])]

            # If none of the opposite pairs have numerator edges → feature=1
            if not any((min(u,v), max(u,v)) in num_edges for u,v in opp_pairs):
                feature_value = 1
                break  # one valid W5 is enough

        results.append([feature_value]*n)

    return results




# Create a seperate dictionary for the eigenvector features
#eigenvector_dict = {f'eigen_{i}': compute_top_k_eigenvector(k=k, i=i) for i in range(k)}

# Dictionary mapping feature names to their computation functions
FEATURE_FUNCTIONS = {
    **eigenvector_functions,  # Add eigenvector functions 
    'adjacency_columns': adjacency_column_features,
    'degree': compute_degree_features,
    'betweenness': compute_betweenness_features,
    'closeness': compute_closeness_features,
    'pagerank': compute_pagerank_features,
    'face_count': compute_face_count_features,
    'W5_indicator': compute_W5_features
}
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



def compute_and_save_feature(feature_name, loop_order, chunk_size=1000, n_jobs=4):
    """
    Compute a single feature for all graphs in a loop order.
    Saves as numpy array.
    """
    # Create output directory
    base = BASE_DIR if 'BASE_DIR' in globals() else Path('../Graph_Edge_Data')
    output_dir = base / f'f_features_loop_{loop_order}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already computed
    output_file = output_dir / f'{feature_name}.npy'
    if output_file.exists():
        print(f"Feature {feature_name} already computed for loop {loop_order}. Skipping.")
        return
    
    print(f"Computing {feature_name} for loop {loop_order}...")
    
    # Load graph edges
    edges_list, labels = load_graph_edges(loop_order)
    n_graphs = len(edges_list)
    
    # Find max nodes considering both types of edges for padding
    max_nodes = max(len(set([u for u,v,_ in edges] + [v for u,v,_ in edges])) for edges in edges_list)
    print(f"Max nodes in graphs: {max_nodes}")
    
    # Get computation function
    if feature_name not in FEATURE_FUNCTIONS:
        raise ValueError(f"Unknown feature: {feature_name}")
    
    compute_func = FEATURE_FUNCTIONS[feature_name]
    
    # Process in chunks
    all_features = []
    
    for i in tqdm(range(0, n_graphs, chunk_size)):
        chunk_edges = edges_list[i:i+chunk_size]
        
        # Compute features in parallel for this chunk
        if n_jobs > 1:
            # Split chunk further for parallel processing
            sub_chunks = [chunk_edges[j:j+100] for j in range(0, len(chunk_edges), 100)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_func)(sub_chunk) for sub_chunk in sub_chunks
            )
            # Flatten results
            chunk_features = [f for sublist in results for f in sublist]
        else:
            chunk_features = compute_func(chunk_edges)
        
        all_features.extend(chunk_features)
    
    # Pad features to consistent size
    features_array = pad_features(all_features, max_nodes)
    
    # Save
    np.save(output_file, features_array)
    print(f"Saved {feature_name} features to {output_file}")
    print(f"Shape: {features_array.shape}")
    
    # Update computed features list
    computed_file = output_dir / 'computed_features.txt'
    with open(computed_file, 'a') as f:
        f.write(f"{feature_name}\n")


def compute_all_features(loop_order, chunk_size=1000, n_jobs=4):
    """Compute all available features for a loop order."""
    print(f"Computing all features for loop {loop_order}...")
    
    for feature_name in FEATURE_FUNCTIONS.keys():
        compute_and_save_feature(feature_name, loop_order, chunk_size, n_jobs)
    
    print("All features computed!")


if __name__ == "__main__":
    # Example usage
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--loop', type=int, help='Loop order (overrides config)')
    parser.add_argument('--feature', type=str, help='Specific feature to compute')
    parser.add_argument('--chunk-size', type=int, help='Chunk size (overrides config)')
    parser.add_argument('--n-jobs', type=int, help='Number of jobs (overrides config)')
    args = parser.parse_args()
    
    config = {
        'data': {
            'loop_order': None,
            'base_dir': './data'
        },
        'features': {
            'chunk_size': 100,
            'n_jobs': 1,
            'compute_all': False,
            'features_to_compute': []
        }
    }

    # Load config if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
        for section in user_config:
            if section in config:
                config[section].update(user_config[section])
            else:
                config[section] = user_config[section]
        
           # Determine loop_order
    if args.loop is not None:
        loop_order_list = [args.loop]  # single integer from CLI
    elif config['data'].get('loop_order') is not None:
        loop_order_list = config['data']['loop_order']
        if isinstance(loop_order_list, int):
            loop_order_list = [loop_order_list]
    else:
        # fallback to train_loop_order or test_loop_order from config
        loop_order_list = config['data'].get('train_loop_order',
                                             config['data'].get('test_loop_order', [1]))
        if isinstance(loop_order_list, int):
            loop_order_list = [loop_order_list]
        
        # Update data path to use base_dir from config
        global BASE_DIR
        BASE_DIR = Path(config['data']['base_dir'])
    chunk_size = args.chunk_size if args.chunk_size is not None else config['features']['chunk_size']
    n_jobs = args.n_jobs if args.n_jobs is not None else config['features']['n_jobs']    
         # Iterate over all loops
    for loop in loop_order_list:
        if args.feature:
            print(f"Computing feature {args.feature} for loop {loop}...")
            compute_and_save_feature(args.feature, loop, chunk_size, n_jobs)
        else:
            if config['features']['compute_all']:
                compute_all_features(loop, chunk_size, n_jobs)
            else:
                for feature in config['features']['features_to_compute']:
                    compute_and_save_feature(feature, loop, chunk_size, n_jobs)
