#!/usr/bin/env python3
import numpy as np
import pandas as pd
import networkx as nx
import ast
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import eigh

# ===============================================================
# Load graph edges (reads from .npz if available)
# ===============================================================

def load_graph_edges(file_ext='7', base_dir=None):
    """
    Load graph edges and labels from fast .npz file if available.
    Encoding:
       -1 : single denominator edge
        1 : single numerator edge
        2 : double numerator edge
        3 : triple numerator edge, etc.
    """
    if base_dir is None:
        base_dir = Path('../Graph_Edge_Data')

    npz_path = Path(base_dir) / f'graph_edges_{file_ext}.npz'
    csv_path = Path(base_dir) / f'graph_data_{file_ext}.csv'

    if npz_path.exists():
        print(f"✅ Loading preprocessed edge file {npz_path.name}")
        data = np.load(npz_path, allow_pickle=True)
        denom_edges_list = data['denom_edges']
        numer_edges_list = data['numer_edges']
        coeffs = data['coefficients']
    elif csv_path.exists():
        print(f"⚠️ No .npz found for loop {file_ext}, falling back to CSV.")
        df = pd.read_csv(csv_path)
        denom_edges_list = [ast.literal_eval(e) for e in df['DEN_EDGES']]
        numer_edges_list = [ast.literal_eval(e) for e in df['NUM_EDGES']]
        from fractions import Fraction
        coeffs = np.array([float(Fraction(str(x))) for x in df["COEFFICIENTS"]], dtype=float)
    else:
        raise FileNotFoundError(f"Neither {npz_path} nor {csv_path} found for loop {file_ext}.")

    labels = [1 if c != 0 else 0 for c in coeffs]

    edge_lists = []
    for d_edges, n_edges in zip(denom_edges_list, numer_edges_list):
        edge_dict = {}
        for u, v in d_edges:
            key = tuple(sorted((u, v)))
            edge_dict[key] = -1
        for u, v in n_edges:
            key = tuple(sorted((u, v)))
            if key in edge_dict and edge_dict[key] == -1:
                raise ValueError(f"Conflict: both numerator and denominator edge between {key}")
            edge_dict[key] = edge_dict.get(key, 0) + 1
        edges = [(u, v, etype) for (u, v), etype in edge_dict.items()]
        edge_lists.append(edges)
    return edge_lists, labels

# ===============================================================
# Helper: convert edges to networkx
# ===============================================================

def edges_to_networkx(edges):
    nodes = sorted(set([u for u, v, _ in edges] + [v for u, v, _ in edges]))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for u, v, etype in edges:
        G.add_edge(node_to_idx[u], node_to_idx[v], edge_type=etype)
    return G, len(nodes)

# ===============================================================
# Feature functions
# ===============================================================

k = 3

def compute_ith_eigenvector(graphs_batch, k=k, i=0):
    features = []
    for edges in graphs_batch:
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)
        A_denom = np.zeros((n, n))
        A_num   = np.zeros((n, n))
        for u, v, etype in edges:
            i_u, i_v = node_to_idx[u], node_to_idx[v]
            if etype == 0 or etype == -1:
                A_denom[i_u, i_v] = A_denom[i_v, i_u] = 1
            elif etype >= 1:
                A_num[i_u, i_v] = A_num[i_v, i_u] = 1
        L_denom = np.diag(A_denom.sum(axis=1)) - A_denom
        L_num   = np.diag(A_num.sum(axis=1))   - A_num
        L_diff  = L_denom - L_num
        eigenvalues, eigenvectors = eigh(L_diff)
        idx = np.argsort(np.real(eigenvalues))[::-1]
        eigenvectors = np.real(eigenvectors[:, idx])
        features.append(eigenvectors[:, i-1].tolist())
    return features

def create_eigen_function(i):
    def eigen_function(graphs_batch):
        return compute_ith_eigenvector(graphs_batch, k=k, i=i)
    return eigen_function

eigenvector_functions = {f'eigen_{i}': create_eigen_function(i) for i in range(1, k+1)}

def compute_degree_features(graphs_batch):
    feats = []
    for edges in graphs_batch:
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        deg = np.zeros(len(nodes))
        for u,v,etype in edges:
            if etype == -1:
                i,j = node_to_idx[u], node_to_idx[v]
                deg[i] += 1; deg[j] += 1
        feats.append(deg)
    return feats

def adjacency_column_features(graphs_batch):
    feats = []
    for edges in graphs_batch:
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node: j for j, node in enumerate(nodes)}
        n = len(nodes)
        A_denom = np.zeros((n,n)); A_num = np.zeros((n,n))
        for u,v,etype in edges:
            i,j = node_to_idx[u], node_to_idx[v]
            if etype == -1: A_denom[i,j]=A_denom[j,i]=1
            elif etype>=1: A_num[i,j]=A_num[j,i]=1
        A_diff = A_denom - A_num
        feats.append([A_diff[:,i].tolist() for i in range(n)])
    return feats

def compute_pagerank_features(graphs_batch):
    feats=[]
    for edges in graphs_batch:
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx = {node:i for i,node in enumerate(nodes)}
        G=nx.Graph(); G.add_nodes_from(range(len(nodes)))
        for u,v,etype in edges:
            if etype==-1:
                i,j=node_to_idx[u],node_to_idx[v]
                G.add_edge(i,j)
        pr=nx.pagerank(G)
        feats.append([pr[i] for i in range(len(nodes))])
    return feats

def compute_betweenness_features(graphs_batch):
    feats=[]
    for edges in graphs_batch:
        nodes = sorted(set([u for u,v,_ in edges] + [v for u,v,_ in edges]))
        node_to_idx={node:i for i,node in enumerate(nodes)}
        G=nx.Graph(); G.add_nodes_from(range(len(nodes)))
        for u,v,etype in edges:
            if etype==-1:
                i,j=node_to_idx[u],node_to_idx[v]
                G.add_edge(i,j)
        bc=nx.betweenness_centrality(G)
        feats.append([bc[i] for i in range(len(nodes))])
    return feats

FEATURE_FUNCTIONS = {
    **eigenvector_functions,
    'adjacency_columns': adjacency_column_features,
    'degree': compute_degree_features,
    'pagerank': compute_pagerank_features,
    'betweenness': compute_betweenness_features
}

# ===============================================================
# Padding helper
# ===============================================================

def pad_features(features_list, max_nodes):
    padded=[]
    for f in features_list:
        arr=np.array(f)
        if arr.ndim==1:
            arr=arr[:,None]
        if arr.shape[0]<max_nodes:
            pad=np.zeros((max_nodes-arr.shape[0],arr.shape[1]))
            arr=np.vstack([arr,pad])
        elif arr.shape[0]>max_nodes:
            arr=arr[:max_nodes]
        padded.append(arr)
    return np.array(padded)

# ===============================================================
# Unified computation + bundling
# ===============================================================

def compute_and_bundle_features(file_ext='7', chunk_size=1000, n_jobs=4,
                                base_dir=None, features_to_compute=None,
                                overwrite=False):
    if base_dir is None:
        base_dir = Path('../Graph_Edge_Data')
    base_dir = Path(base_dir)
    folder_name = f"f_features_loop_{file_ext}"
    out_dir = base_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir.with_suffix(".npz")

    edges_list, labels = load_graph_edges(file_ext, base_dir)
    n_graphs = len(edges_list)
    max_nodes = max(len(set([u for u,v,_ in e]+[v for u,v,_ in e])) for e in edges_list)
    print(f"📊 {n_graphs} graphs loaded (max nodes {max_nodes})")

    feature_arrays = {}
    for name in features_to_compute:
        npy_path = out_dir / f"{name}.npy"
        if npz_path.exists() and not overwrite:
            with np.load(npz_path, allow_pickle=True) as zf:
                if name in zf.files:
                    print(f"✔ {name} already in {npz_path.name}, skipping.")
                    feature_arrays[name]=zf[name]; continue
        if npy_path.exists() and not overwrite:
            print(f"✔ {name}.npy exists, loading.")
            feature_arrays[name]=np.load(npy_path); continue
        if name not in FEATURE_FUNCTIONS:
            print(f"⚠ Unknown feature {name}, skipping."); continue

        func=FEATURE_FUNCTIONS[name]
        print(f"🧮 Computing {name} ...")
        all_feats=[]
        for i in tqdm(range(0,n_graphs,chunk_size)):
            chunk=edges_list[i:i+chunk_size]
            if n_jobs>1:
                sub=[chunk[j:j+100] for j in range(0,len(chunk),100)]
                results=Parallel(n_jobs=n_jobs)(
                    delayed(func)(sub_chunk) for sub_chunk in sub)
                chunk_feats=[f for sublist in results for f in sublist]
            else:
                chunk_feats=func(chunk)
            all_feats.extend(chunk_feats)
        arr=pad_features(all_feats,max_nodes)
        np.save(npy_path,arr)
        feature_arrays[name]=arr
        print(f"💾 Saved {name} → {npy_path}")

    print(f"📦 Bundling features → {npz_path.name}")
    np.savez_compressed(npz_path, **feature_arrays)
    print(f"✅ Saved bundle: {npz_path} ({npz_path.stat().st_size/1e6:.2f} MB)")

# ===============================================================
# Main entry
# ===============================================================

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--file_ext', type=str, help='Loop order override')
    parser.add_argument('--overwrite', action='store_true', help='Recompute even if files exist')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    file_ext = args.file_ext or config['data'].get('file_ext', '7')
    base_dir = config['data'].get('base_dir', '../Graph_Edge_Data')
    chunk_size = config['features'].get('chunk_size', 1000)
    n_jobs = config['features'].get('n_jobs', 4)
    feats = config['features'].get('features_to_compute', [])

    if not feats:
        print("⚠ No features listed in config['features_to_compute']. Exiting.")
    else:
        compute_and_bundle_features(file_ext, chunk_size, n_jobs, base_dir, feats, overwrite=args.overwrite)
