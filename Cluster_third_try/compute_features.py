import numpy as np
import pandas as pd
import networkx as nx
import ast
from pathlib import Path
from tqdm import tqdm
import os
from joblib import Parallel, delayed
from scipy.linalg import eigvals


def load_graph_edges(loop_order, data_path=None):
    """Load graph edges from CSV file."""
    if data_path is None:
        # Use BASE_DIR if defined, otherwise default path
        base = BASE_DIR if 'BASE_DIR' in globals() else Path('Graph_Edge_Data')
        data_path = base / f'den_graph_data_{loop_order}.csv'
    
    df = pd.read_csv(data_path)
    edges = [ast.literal_eval(e) for e in df['EDGES']]
    labels = df['COEFFICIENTS'].tolist()
    
    return edges, labels


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

def compute_laplacian_eigenvalues_normalized(graphs_batch):
    """Compute normalized Laplacian eigenvalues for a batch of graphs."""
    features = []

    for edges in graphs_batch:
        G, n_nodes= edges_to_networkx(edges)
        
        # Get the normalized Laplacian matrix
        normalized_laplacian = nx.normalized_laplacian_matrix(G).toarray()
        
        # Compute eigenvalues
        eigenvalues = eigvals(normalized_laplacian)
        
        # Sort eigenvalues in ascending order
        eigenvalues = np.sort(np.real(eigenvalues))
        
        features.append(eigenvalues.tolist())
    
    return features

def compute_degree_features(graphs_batch):
    """Compute degree for a batch of graphs."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        degrees = [G.degree(i) for i in range(n_nodes)]
        features.append(degrees)
    
    return features


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


def compute_face_count_features(graphs_batch):
    """Compute number of faces each node belongs to (planar graph feature)."""
    features = []
    
    for edges in graphs_batch:
        G, n_nodes = edges_to_networkx(edges)
        
        # Get planar embedding
        is_planar, embedding = nx.check_planarity(G)
        if not is_planar:
            # Fallback: use degree as proxy
            face_counts = [G.degree(i) for i in range(n_nodes)]
        else:
            # Count faces for each node
            face_counts = [0] * n_nodes
            visited_edges = set()
            
            for node in embedding.nodes():
                for neighbor in embedding.neighbors(node):
                    edge = (node, neighbor) if node < neighbor else (neighbor, node)
                    if edge not in visited_edges:
                        face = list(embedding.traverse_face(node, neighbor))
                        for v in face:
                            if v < n_nodes:  # Safety check
                                face_counts[v] += 1
                        # Mark edges as visited
                        for i in range(len(face)):
                            u, v = face[i], face[(i + 1) % len(face)]
                            visited_edges.add((u, v) if u < v else (v, u))
        
        features.append(face_counts)
    
    return features


# Dictionary mapping feature names to their computation functions
FEATURE_FUNCTIONS = {
    'laplacian_eigenvalues_normalized': compute_laplacian_eigenvalues_normalized,
    'degree': compute_degree_features,
    'betweenness': compute_betweenness_features,
    'clustering': compute_clustering_features,
    'closeness': compute_closeness_features,
    'pagerank': compute_pagerank_features,
    'face_count': compute_face_count_features
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


def compute_and_save_feature(feature_name, loop_order, chunk_size=1000, n_jobs=4):
    """
    Compute a single feature for all graphs in a loop order.
    Saves as numpy array.
    """
    # Create output directory
    base = BASE_DIR if 'BASE_DIR' in globals() else Path('Graph_Edge_Data')
    output_dir = base / f'features_loop_{loop_order}'
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
    
    # Find max nodes for padding
    max_nodes = max(len(set(u for e in edges for u in e)) for edges in edges_list)
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
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--loop', type=int, help='Loop order (overrides config)')
    parser.add_argument('--feature', type=str, help='Specific feature to compute')
    parser.add_argument('--chunk-size', type=int, help='Chunk size (overrides config)')
    parser.add_argument('--n-jobs', type=int, help='Number of jobs (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config, with command line overrides
    loop_order = args.loop if args.loop is not None else config['data']['loop_order']
    chunk_size = args.chunk_size if args.chunk_size is not None else config['features']['chunk_size']
    n_jobs = args.n_jobs if args.n_jobs is not None else config['features']['n_jobs']
    
    # Update data path to use base_dir from config
    global BASE_DIR
    BASE_DIR = Path(config['data']['base_dir'])
    
    if args.feature:
        compute_and_save_feature(args.feature, loop_order, chunk_size, n_jobs)
    else:
        # Use features list from config
        if config['features']['compute_all']:
            compute_all_features(loop_order, chunk_size, n_jobs)
        else:
            for feature in config['features']['features_to_compute']:
                compute_and_save_feature(feature, loop_order, chunk_size, n_jobs)
