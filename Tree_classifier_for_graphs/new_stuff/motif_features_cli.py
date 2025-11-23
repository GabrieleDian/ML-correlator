#!/usr/bin/env python3
# motif_features_cli.py
# -------------------------------------------------------------
# Motif-based feature extractor for graph analysis
# Extracts only motif-related features from graphs including:
# - 3-motifs: triangles, wedges
# - 4-motifs: 4-cycles, 4-cliques
# - 5-motifs: 5-cycles, 5-cliques
# - Induced 4-node subgraphs: g_1^4 to g_6^4 (Path4, Star4, Cycle4, TailedTriangle, Diamond, Clique4)
# - Induced 5-node subgraphs: g_1^5 to g_11^5 (all connected 5-node graphlets)
# - Cross-layer motifs (when both DEN & NUM layers exist)
# - Edge triangle incidence statistics
# - Square clustering proxy
#
# Input CSV:
#   - Either EDGES (single layer) OR DEN_EDGES + NUM_EDGES (two-layer)
#   - Edge lists must be Python-like, e.g. "[(0,1),(1,2)]"
#   - Optional COEFFICIENTS column is preserved (can be kept first)
#
# Examples:
#   Extract all motif features:
#     python motif_features_cli.py --input data.csv --output motif_feats.parquet
#     python motif_features_cli.py --input data.csv --output motif_feats.csv
#
#   Extract only 3/4-motifs:
#     python motif_features_cli.py --input data.csv --output motif_feats.parquet --groups motifs34
#
#   Extract only induced subgraphs:
#     python motif_features_cli.py --input data.csv --output motif_feats.csv --groups induced4,induced5
#
#   Extract only spectral features:
#     python motif_features_cli.py --input data.csv --output motif_feats.csv --groups spectral
#
#   Parallel processing:
#     python motif_features_cli.py --input data.csv --output motif_feats.parquet --workers 8 --intra-workers 4
# -------------------------------------------------------------

import os, argparse, ast, warnings, math, random, json
from collections import Counter, defaultdict
from itertools import combinations
from math import log2
from typing import Dict, Any, Iterable, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# ---------------- perf & warnings ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation.*", category=RuntimeWarning)
try:
    from scipy._lib._warnings import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except Exception:
    pass
try:
    from scipy.stats import ConstantInputWarning as SciPyConstantInputWarning
    warnings.filterwarnings("ignore", category=SciPyConstantInputWarning)
except Exception:
    pass

# --------------- optional deps -------------------
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    from tqdm_joblib import tqdm_joblib
    _HAS_TQDM_JOBLIB = True
except Exception:
    _HAS_TQDM_JOBLIB = False

# WL hashing / Atlas for induced 5-node graphlets
try:
    from networkx.generators.atlas import graph_atlas_g
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash
    _HAS_WL = True
except Exception:
    _HAS_WL = False

# ========================= Feature groups =========================
# Available feature groups for motif extraction
ALL_GROUPS = [
    "motifs34",      # 3-motifs (triangles, wedges) and 4-motifs (4-cycles, 4-cliques)
    "motifs5",       # 5-motifs (5-cycles, 5-cliques)
    "induced4",      # Induced 4-node subgraphs (g_1^4 to g_6^4)
    "induced5",      # Induced 5-node subgraphs (g_1^5 to g_11^5)
    "cross_layer",   # Cross-layer motifs (when both DEN and NUM layers exist)
    "spectral"       # Spectral features (Laplacian eigenvalues/eigenvectors, adjacency spectrum)
]

# ========================= Utility functions =========================
def _detect_output_format(output_file):
    """Detect output format based on file extension.
    
    Args:
        output_file: Path to output file
    
    Returns:
        str: 'csv' or 'parquet'
    """
    ext = os.path.splitext(output_file)[1].lower()
    if ext == '.csv':
        return 'csv'
    elif ext == '.parquet':
        return 'parquet'
    else:
        # Default to parquet for unknown extensions
        return 'parquet'

def _save_dataframe(df, output_file, format_type=None):
    """Save DataFrame to file in the specified format.
    
    Args:
        df: pandas DataFrame to save
        output_file: Path to output file
        format_type: 'csv' or 'parquet' (auto-detected if None)
    """
    if format_type is None:
        format_type = _detect_output_format(output_file)
    
    if format_type == 'csv':
        df.to_csv(output_file, index=False)
    elif format_type == 'parquet':
        df.to_parquet(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def _q(vals, q):
    """Compute quantile of values, returning NaN for empty lists.
    
    Args:
        vals: List of numeric values
        q: Quantile value (0.0 to 1.0)
    
    Returns:
        float: Quantile value or NaN if vals is empty
    """
    if not vals or len(vals) == 0:
        return np.nan
    return float(np.quantile(vals, q))

def _frac(vals, pred):
    """Compute fraction of values satisfying a predicate.
    
    Args:
        vals: List of values to test
        pred: Predicate function to test each value
    
    Returns:
        float: Fraction of values satisfying predicate, or NaN if vals is empty
    """
    if not vals or len(vals) == 0:
        return np.nan
    return float(sum(1 for v in vals if pred(v)) / len(vals))

def try_or_nan(fn, *args, **kwargs):
    """Execute function with given arguments, returning NaN on any exception.
    
    Args:
        fn: Function to execute
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Function result or NaN if exception occurs
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return np.nan

# ========================= Spectral helper functions =========================
def _laplacian_eigs(G):
    """Compute sorted Laplacian eigenvalues of a graph.
    
    Args:
        G: NetworkX graph
    
    Returns:
        np.array: Sorted Laplacian eigenvalues (ascending order)
    """
    if G.number_of_nodes() == 0:
        return np.array([])
    L = nx.laplacian_matrix(G).todense()
    return np.sort(np.linalg.eigvalsh(L))

def _laplacian_heat_traces(leigs, ts):
    """Compute Laplacian heat traces for given time points.
    
    Args:
        leigs: Laplacian eigenvalues
        ts: Iterable of time points
    
    Returns:
        dict: Heat trace values for each time point
    """
    traces = {}
    if leigs is None or len(leigs) == 0 or np.any(np.isnan(leigs)):
        for t in ts: 
            traces[f"Spectral_laplacian_heat_trace_t{t}"] = np.nan
        return traces
    for t in ts:
        traces[f"Spectral_laplacian_heat_trace_t{t}"] = float(np.sum(np.exp(-t * leigs)))
    return traces

def _adjacency_spectrum_features(A):
    """Compute adjacency matrix spectrum features.
    
    Args:
        A: Adjacency matrix
    
    Returns:
        tuple: (energy, estrada_index, moment_2, moment_3, moment_4)
    """
    if A.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    evals = np.linalg.eigvalsh(A)
    energy  = float(np.sum(np.abs(evals)))
    estrada = float(np.sum(np.exp(evals)))
    n = len(evals) if len(evals) > 0 else np.nan
    m2 = float(np.mean(evals**2)) if n == n and n > 0 else np.nan
    m3 = float(np.mean(evals**3)) if n == n and n > 0 else np.nan
    m4 = float(np.mean(evals**4)) if n == n and n > 0 else np.nan
    return energy, estrada, m2, m3, m4

# ========================= Motif helpers =========================
def _adjacency_matrix(G):
    """Convert NetworkX graph to adjacency matrix.
    
    Args:
        G: NetworkX graph
    
    Returns:
        numpy.ndarray: Adjacency matrix with nodes in sorted order
    """
    return nx.to_numpy_array(G, dtype=float, nodelist=list(G.nodes()))

def count_triangles(G, A=None):
    """Count triangles in the graph using matrix multiplication.
    
    Uses the fact that trace(A^3) = 6 * number_of_triangles for undirected graphs.
    This is more efficient than enumerating all triangles for large graphs.
    
    Args:
        G: NetworkX graph
        A: Precomputed adjacency matrix (optional)
    
    Returns:
        int: Number of triangles in the graph
    
    Also known as: 3-cliques, K3, g_1^3
    """
    if A is None:
        A = _adjacency_matrix(G)
    A3 = A @ A @ A
    return int(round(np.trace(A3) / 6.0))

def count_wedges(G, degs=None, tri=None):
    """Count wedges (2-paths) in the graph.
    
    A wedge is a path of length 2 (two edges sharing a common vertex).
    Uses the formula: wedges = sum(d*(d-1)/2) - 3*triangles
    In code below - we have triangle count precomputed for efficiency.
    
    Args:
        G: NetworkX graph
        degs: Precomputed degree sequence (optional)
        tri: Precomputed triangle count (optional)
    
    Returns:
        int: Number of wedges in the graph
    
    Also known as: 2-paths, g_2^3
    """
    if degs is None:
        degs = [d for _, d in G.degree()]
    if tri is None:
        tri = count_triangles(G)
    s = sum(d*(d-1)//2 for d in degs)
    return int(s - 3*tri)

def count_4_cycles(G, A=None):
    """Count 4-cycles in the graph.
    
    Uses matrix multiplication: A^2[i,j] gives the number of paths of length 2
    between nodes i and j. For each pair (i,j) with at least 2 paths, we count
    the number of ways to choose 2 paths to form a 4-cycle.
    
    Args:
        G: NetworkX graph
        A: Precomputed adjacency matrix (optional)
    
    Returns:
        int: Number of 4-cycles in the graph
    
    Also known as: C4, g_3^4
    example: v1​−v2​−v3​−v4​−v1​
    """
    if A is None:
        A = _adjacency_matrix(G)
    A2 = A @ A
    n = A2.shape[0]
    total_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            cij = int(round(A2[i, j]))
            if cij >= 2:
                total_pairs += (cij * (cij - 1)) // 2
    return int(total_pairs // 2)

def count_4_cliques(G):
    """Count 4-cliques in the graph.
    
    Uses NetworkX's clique enumeration algorithm. For each clique of size k >= 4,
    counts the number of 4-cliques it contains using binomial coefficient C(k,4).
    
    Args:
        G: NetworkX graph
    
    Returns:
        int: Number of 4-cliques in the graph
    
    Also known as: K4, g_6^4
    """
    cnt = 0
    for clq in nx.find_cliques(G):
        k = len(clq)
        if k >= 4:
            cnt += math.comb(k, 4)
    return int(cnt)

def count_5_cycles(G):
    """Count 5-cycles in the graph.
    
    Uses exhaustive enumeration of all possible 5-cycles. For each ordered sequence
    of 5 nodes (a,b,c,d,e), checks if it forms a cycle: a-b-c-d-e-a.
    Uses neighbor sets for efficient checking and avoids duplicates.
    
    Args:
        G: NetworkX graph
    
    Returns:
        int: Number of 5-cycles in the graph
    
    Also known as: C5, g_7^5 (Pentagon-shaped 5-cycle)
    """
    if G.number_of_nodes() < 5:
        return 0
    count = 0
    nodes = sorted(G.nodes())
    nbrs = {u: set(G[u]) for u in nodes}
    for a in nodes:
        for b in sorted(nbrs[a]):
            if b <= a: continue
            for c in sorted(nbrs[b]):
                if c == a or c <= a or c == b: continue
                for d in sorted(nbrs[c]):
                    if d in (a,b,c) or d <= a: continue
                    for e in sorted(nbrs[d]):
                        if e in (a,b,c,d) or e <= a: continue
                        if a in nbrs[e] and b < e:
                            count += 1
    return int(count)

def count_5_cliques(G):
    """Count 5-cliques in the graph.
    
    Uses NetworkX's clique enumeration algorithm. For each clique of size k >= 5,
    counts the number of 5-cliques it contains using binomial coefficient C(k,5).
    
    Args:
        G: NetworkX graph
    
    Returns:
        int: Number of 5-cliques in the graph
    
    Also known as: K5, g_20^5 (Complete graph K5)
    """
    cnt = 0
    for clq in nx.find_cliques(G):
        k = len(clq)
        if k >= 5:
            cnt += math.comb(k, 5)
    return int(cnt)

def compute_edge_triangle_incidence_stats(G):
    """Compute statistics of triangle edge incidence.
    
    Measures how many triangles each edge participates in.
    
    How tightly the graph is clustered around triangles.
    Computes statistics on how many triangles each edge belongs to.
    """
    out = []
    for u, v in G.edges():
        out.append(len(set(G[u]) & set(G[v])))
    if not out:
        return 0.0, 0.0
    arr = np.array(out, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))

def compute_square_clustering_proxy(G, A=None):
    """Compute square clustering proxy metric.
    
    Measures the tendency to form 4-cycles relative to 2-paths.
    Same as compute_edge_triangle_incidence_stats but towards 4-cycles.
    """
    if A is None:
        A = _adjacency_matrix(G)
    A2 = A @ A
    n = A.shape[0]
    denom = 0
    for i in range(n):
        for j in range(i+1, n):
            cij = int(round(A2[i, j]))
            if cij >= 2:
                denom += (cij * (cij - 1)) // 2
    c4 = count_4_cycles(G, A=A)
    if denom == 0:
        return np.nan
    return float(4.0 * c4 / denom)

# ---- induced 4-node ----
def classify_induced_4node_subgraph(G_sub):
    """Classify a 4-node induced subgraph into a canonical type.
    
    Returns graphlet notation:
    - g_1_4: Path4 (P4) - 4-node path
    - g_2_4: Star4 (K1_3) - 4-node star  
    - g_3_4: Cycle4 (C4) - 4-node cycle
    - g_4_4: TailedTriangle - triangle with tail
    - g_5_4: Diamond - 4-node diamond
    - g_6_4: Clique4 (K4) - 4-node clique

    All connected 4-node graphlets are enumerated.
    """
    if not nx.is_connected(G_sub): return None
    m = G_sub.number_of_edges()
    degs = sorted([d for _, d in G_sub.degree()])
    if m == 3:
        if degs == [1,1,2,2]: return "g_1_4"  # Path4 (P4)
        if degs == [1,1,1,3]: return "g_2_4"  # Star4 (K1_3)
    elif m == 4:
        if degs == [2,2,2,2]: return "g_3_4"  # Cycle4 (C4)
        if degs == [1,2,2,3]: return "g_4_4"  # TailedTriangle
    elif m == 5:
        if degs == [2,2,3,3]: return "g_5_4"  # Diamond
    elif m == 6:
        if degs == [3,3,3,3]: return "g_6_4"  # Clique4 (K4)
    return None

# ========================= induced 5-node (WL) =========================
# 5-node graphlet mappings based on graphlet_mappings_5node.csv:
# g_0^5:  4-leaf star (Center node connected to 4 leaves)
# g_1^5:  3-leaf star + extra edge (One branch extended by an extra node)
# g_2^5:  Path on 5 nodes (P5) (Simple chain of 4 edges)
# g_3^5:  Tailed triangle (Triangle with a 2-node tail)
# g_4^5:  Triangle with a branch (Triangle with a 2-edge branch)
# g_5^5:  Tailed diamond (Diamond (K4−e) plus one leaf)
# g_6^5:  House graph (4-cycle + roof node)
# g_7^5:  5-cycle (C5) (Pentagon-shaped 5-cycle)
# g_8^5:  Triangle + chord (Triangle with an extra edge forming overlap)
# g_9^5:  Diamond + tail (K4−e with one attached node)
# g_10^5: 4-cycle + diagonal (Square with one diagonal)
# g_11^5: House + diagonal (House graph with extra cross edge)
# g_12^5: Double-triangle (Two triangles sharing one vertex)
# g_13^5: Diamond + branch (K4−e plus one extra neighbor)
# g_14^5: Kite graph (Triangle + extra edge between leaves)
# g_15^5: Triangular prism-like (Two triangles joined by edge)
# g_16^5: 4-cycle + central hub (Square with one central node connected to two)
# g_17^5: 5-node almost-complete (K5 missing one edge)
# g_18^5: Bowtie + extra node (Two triangles joined at one node, plus extra)
# g_19^5: Wheel W5 (5-cycle with a central hub)
# g_20^5: Complete graph K5 (All nodes interconnected)
_G5_REPS = []; _G5_HASH2NAME = {}; _G5_NAMES = []
def _init_g5_representatives():
    """Initialize canonical representatives for 5-node graphlets.
    
    This function creates a canonical set of 5-node connected graphs and maps them
    to standard graphlet notation (g_0^5 to g_20^5) based on the graphlet atlas.
    Uses Weisfeiler-Lehman hashing for efficient isomorphism testing.
    
    Global variables updated:
        _G5_REPS: List of canonical 5-node graph representatives
        _G5_HASH2NAME: Dictionary mapping WL hash to graphlet name
        _G5_NAMES: List of graphlet names in canonical order
    
    The mapping follows the standard 5-node graphlet enumeration:
    - g_0^5: 4-leaf star (K1,4)
    - g_1^5: 3-leaf star + extra edge
    - g_2^5: Path on 5 nodes (P5)
    - g_3^5: Tailed triangle
    - g_4^5: Triangle with a branch
    - g_5^5: Tailed diamond
    - g_6^5: House graph
    - g_7^5: 5-cycle (C5)
    - g_8^5: Triangle + chord
    - g_9^5: Diamond + tail
    - g_10^5: 4-cycle + diagonal
    - g_11^5: House + diagonal
    - g_12^5: Double-triangle
    - g_13^5: Diamond + branch
    - g_14^5: Kite graph
    - g_15^5: Triangular prism-like
    - g_16^5: 4-cycle + central hub
    - g_17^5: 5-node almost-complete
    - g_18^5: Bowtie + extra node
    - g_19^5: Wheel W5
    - g_20^5: Complete graph K5
    """
    global _G5_REPS, _G5_HASH2NAME, _G5_NAMES
    if _G5_REPS or not _HAS_WL:
        return
    try:
        atlas = graph_atlas_g()
        cand = [g.copy() for g in atlas if g.number_of_nodes() == 5 and nx.is_connected(g)]
        reps = []
        for g in cand:
            if not any(nx.is_isomorphic(g, r) for r in reps):
                reps.append(nx.convert_node_labels_to_integers(g, ordering="sorted"))
        reps.sort(key=lambda H: (H.number_of_edges(), sorted([d for _, d in H.degree()])))
        
        # Create graphlet names based on the standard 5-node graphlet atlas
        # Mapping follows the canonical ordering from graphlet_mappings_5node.csv
        names = []
        for i, rep in enumerate(reps):
            m = rep.number_of_edges()
            degs = sorted([d for _, d in rep.degree()])
            
            # Map to standard graphlet notation g_i^5 based on structure
            # This follows the canonical 5-node graphlet enumeration
            if m == 4:
                if degs == [1,1,1,1,4]: names.append("g_0_5")  # 4-leaf star
                else: names.append(f"g_{i}_5")
            elif m == 5:
                if degs == [1,1,1,2,3]: names.append("g_1_5")  # 3-leaf star + extra edge
                elif degs == [1,1,2,2,2]: names.append("g_2_5")  # Path on 5 nodes (P5)
                else: names.append(f"g_{i}_5")
            elif m == 6:
                if degs == [1,1,2,2,2]: names.append("g_3_5")  # Tailed triangle
                elif degs == [1,2,2,2,3]: names.append("g_4_5")  # Triangle with a branch
                elif degs == [2,2,2,2,2]: names.append("g_7_5")  # 5-cycle (C5)
                else: names.append(f"g_{i}_5")
            elif m == 7:
                if degs == [1,2,2,2,4]: names.append("g_5_5")  # Tailed diamond
                elif degs == [2,2,2,3,3]: names.append("g_6_5")  # House graph
                elif degs == [2,2,2,2,3]: names.append("g_8_5")  # Triangle + chord
                else: names.append(f"g_{i}_5")
            elif m == 8:
                if degs == [2,2,2,3,3]: names.append("g_9_5")  # Diamond + tail
                elif degs == [2,2,3,3,4]: names.append("g_10_5")  # 4-cycle + diagonal
                elif degs == [2,2,3,3,4]: names.append("g_11_5")  # House + diagonal
                elif degs == [2,2,3,3,4]: names.append("g_12_5")  # Double-triangle
                elif degs == [2,2,3,3,4]: names.append("g_13_5")  # Diamond + branch
                elif degs == [2,2,3,3,4]: names.append("g_14_5")  # Kite graph
                elif degs == [2,2,3,3,4]: names.append("g_15_5")  # Triangular prism-like
                elif degs == [2,2,3,3,4]: names.append("g_16_5")  # 4-cycle + central hub
                elif degs == [2,2,3,3,4]: names.append("g_17_5")  # 5-node almost-complete
                elif degs == [2,2,3,3,4]: names.append("g_18_5")  # Bowtie + extra node
                elif degs == [2,2,3,3,4]: names.append("g_19_5")  # Wheel W5
                else: names.append(f"g_{i}_5")
            elif m == 9:
                names.append("g_17_5")  # 5-node almost-complete (K5 missing one edge)
            elif m == 10:
                names.append("g_20_5")  # Complete graph K5
            else:
                names.append(f"g_{i}_5")
        
        hash2name = {}
        for name, rep in zip(names, reps):
            h = wl_hash(rep)
            hash2name[h] = name
        _G5_REPS, _G5_HASH2NAME, _G5_NAMES = reps, hash2name, names
    except Exception:
        _G5_REPS, _G5_HASH2NAME, _G5_NAMES = [], {}, []
try:
    _init_g5_representatives()
except Exception:
    _G5_REPS, _G5_HASH2NAME, _G5_NAMES = [], {}, []

def _chunked(iterable, n_chunks):
    """Split an iterable into approximately equal-sized chunks.
    
    Args:
        iterable: Iterable to split into chunks
        n_chunks: Number of chunks to create
    
    Yields:
        list: Chunks of the original iterable
    
    Examples:
        >>> list(_chunked([1,2,3,4,5], 2))
        [[1, 2, 3], [4, 5]]
        >>> list(_chunked([1,2,3,4,5], 3))
        [[1, 2], [3, 4], [5]]
    """
    it = list(iterable)
    if n_chunks <= 1 or len(it) == 0:
        yield it; return
    size = (len(it) + n_chunks - 1) // n_chunks
    for i in range(0, len(it), size):
        yield it[i:i+size]

def _induced4_counts(G, *, max_samples=None, rng=None, n_jobs: int = 0):
    """Count induced 4-node subgraphs in a graph.
    
    Enumerates all 4-node subsets and classifies their induced subgraphs into
    canonical 4-node graphlet types. Uses parallel processing for large graphs.
    
    Args:
        G: NetworkX graph to analyze
        max_samples: Maximum number of 4-node subsets to sample (None for all)
        rng: Random number generator for sampling
        n_jobs: Number of parallel jobs (0 for sequential processing)
    
    Returns:
        dict: Counts of each 4-node graphlet type:
            - g_1_4: Path4 (P4) - 4-node path
            - g_2_4: Star4 (K1,3) - 4-node star
            - g_3_4: Cycle4 (C4) - 4-node cycle
            - g_4_4: TailedTriangle - triangle with tail
            - g_5_4: Diamond - 4-node diamond
            - g_6_4: Clique4 (K4) - 4-node clique
    
    Notes:
        - Returns zero counts for graphs with fewer than 4 nodes
        - Uses sampling for very large graphs to avoid combinatorial explosion
        - Parallel processing available via joblib when n_jobs > 0
    """
    counts0 = {"g_1_4": 0, "g_2_4": 0, "g_3_4": 0, "g_4_4": 0, "g_5_4": 0, "g_6_4": 0}  # Path4, Star4, Cycle4, TailedTriangle, Diamond, Clique4
    nodes = list(G.nodes()); n = len(nodes)
    if n < 4:
        return counts0

    if max_samples is None:
        combos = list(combinations(nodes, 4))
    else:
        rng = rng or random.Random(42)
        total = math.comb(n, 4)
        k = min(max_samples, total)
        seen = set()
        while len(seen) < k:
            S = tuple(sorted(rng.sample(nodes, 4)))
            if S not in seen:
                seen.add(S)
        combos = list(seen)

    def _accumulate(part):
        loc = counts0.copy()
        for S in part:
            H = G.subgraph(S)
            key = classify_induced_4node_subgraph(H)
            if key is not None: loc[key] += 1
        return loc

    if _HAS_JOBLIB and n_jobs and n_jobs > 1 and len(combos) > 5000:
        parts = list(_chunked(combos, n_jobs))
        results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_accumulate)(p) for p in parts)
        out = counts0.copy()
        for r in results:
            for k in out: out[k] += r[k]
        return out
    else:
        return _accumulate(combos)

def _induced5_counts(G, *, sample_fraction: float = 1.0, random_state: int = 42, n_jobs: int = 0):
    """Count induced 5-node subgraphs in a graph.
    
    Enumerates 5-node subsets and classifies their induced subgraphs into
    canonical 5-node graphlet types using Weisfeiler-Lehman hashing.
    Uses sampling and parallel processing for efficiency.
    
    Args:
        G: NetworkX graph to analyze
        sample_fraction: Fraction of 5-node subsets to sample (0.0 to 1.0)
        random_state: Random seed for reproducible sampling
        n_jobs: Number of parallel jobs (0 for sequential processing)
    
    Returns:
        tuple: (counts_dict, total_connected)
            - counts_dict: Counts of each 5-node graphlet type (g_0^5 to g_20^5)
            - total_connected: Total number of connected 5-node subgraphs
    
    Notes:
        - Returns zero counts for graphs with fewer than 5 nodes
        - Requires Weisfeiler-Lehman hashing for isomorphism testing
        - Uses sampling to handle large graphs efficiently
        - Parallel processing available via joblib when n_jobs > 0
        - Graphlet names follow canonical 5-node enumeration
    """
    counts = {name: 0 for name in _G5_NAMES}
    if G.number_of_nodes() < 5 or not _G5_NAMES or not _HAS_WL:
        return counts, float("nan")

    nodes = list(G.nodes())
    combos = list(combinations(nodes, 5))
    if sample_fraction < 1.0:
        rnd = random.Random(random_state)
        combos = [c for c in combos if rnd.random() < sample_fraction]

    def _accumulate(part):
        loc = {name: 0 for name in _G5_NAMES}
        connected_total = 0
        for S in part:
            H = G.subgraph(S)
            if not nx.is_connected(H): 
                continue
            connected_total += 1
            try:
                h = wl_hash(H)
                name = _G5_HASH2NAME.get(h)
                if name is None:
                    for rep, rep_name in zip(_G5_REPS, _G5_NAMES):
                        if nx.is_isomorphic(H, rep):
                            name = rep_name; _G5_HASH2NAME[h] = name; break
                if name is not None:
                    loc[name] += 1
            except Exception:
                continue
        return loc, connected_total

    if _HAS_JOBLIB and n_jobs and n_jobs > 1 and len(combos) > 2000:
        parts = list(_chunked(combos, n_jobs))
        results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_accumulate)(p) for p in parts)
        out = {name: 0 for name in _G5_NAMES}; conn = 0
        for loc, c in results:
            conn += c
            for k in out: out[k] += loc[k]
        return out, float(conn)
    else:
        return _accumulate(combos)

# ========================= Cross-layer motif functions =========================
def count_cross_layer_motifs(G_den, G_num, motif="triangle"):
    G_total = nx.compose(G_den, G_num)
    mixed = 0
    if motif == "triangle":
        for clq in nx.enumerate_all_cliques(G_total):
            if len(clq) == 3:
                edges = [(clq[i], clq[j]) for i in range(3) for j in range(i+1, 3)]
                layers = []
                for u, v in edges:
                    if G_den.has_edge(u, v): layers.append("DEN")
                    elif G_num.has_edge(u, v): layers.append("NUM")
                if len(set(layers)) > 1: mixed += 1
    elif motif == "4cycle":
        total_c4 = count_4_cycles(G_total)
        c4_den   = count_4_cycles(G_den)
        c4_num   = count_4_cycles(G_num)
        mixed = max(total_c4 - (c4_den + c4_num), 0)
    elif motif in ("diamond", "4clique"):
        for nodes in combinations(G_total.nodes(), 4):
            H = G_total.subgraph(nodes)
            if not nx.is_connected(H): continue
            m = H.number_of_edges(); degs = sorted([d for _, d in H.degree()])
            if motif == "diamond" and (m == 5 and degs == [2,2,3,3]):
                edges = list(H.edges())
            elif motif == "4clique" and m == 6:
                edges = list(H.edges())
            else:
                continue
            layers = []
            for u, v in edges:
                if G_den.has_edge(u, v): layers.append("DEN")
                elif G_num.has_edge(u, v): layers.append("NUM")
            if len(set(layers)) > 1:
                mixed += 1
    return mixed

# ========================= Graph building =========================
def build_graphs_from_row(row):
    """Build NetworkX graphs from a CSV row containing edge data.
    
    Handles both single-layer (EDGES) and two-layer (DEN_EDGES, NUM_EDGES) formats.
    Edge lists should be Python-evaluable strings like "[(0,1),(1,2)]".
    
    Args:
        row: pandas Series containing edge data
    
    Returns:
        dict: Dictionary with graph objects:
            - "SINGLE": Single graph (if EDGES present)
            - "DEN": DEN layer graph (if DEN_EDGES present)
            - "NUM": NUM layer graph (if NUM_EDGES present)
            - "TOTAL": Combined graph (if both DEN and NUM present)
    """
    graphs = {}
    
    # Single layer (EDGES)
    if "EDGES" in row and pd.notna(row["EDGES"]):
        try:
            edges = ast.literal_eval(row["EDGES"])
            G = nx.Graph(edges)
            return {"SINGLE": G}
        except Exception:
            pass
    
    # Two layers (DEN_EDGES, NUM_EDGES)
    G_DEN = nx.Graph()
    G_NUM = nx.Graph()
    
    if "DEN_EDGES" in row and pd.notna(row["DEN_EDGES"]):
        try:
            edges = ast.literal_eval(row["DEN_EDGES"])
            G_DEN = nx.Graph(edges)
        except Exception:
            pass
    
    if "NUM_EDGES" in row and pd.notna(row["NUM_EDGES"]):
        try:
            edges = ast.literal_eval(row["NUM_EDGES"])
            G_NUM = nx.Graph(edges)
        except Exception:
            pass
    
    # Return appropriate graphs
    if G_DEN.number_of_edges() > 0 and G_NUM.number_of_edges() > 0:
        G_TOTAL = nx.compose(G_DEN, G_NUM)
        return {"DEN": G_DEN, "NUM": G_NUM, "TOTAL": G_TOTAL}
    elif G_DEN.number_of_edges() > 0:
        return {"DEN": G_DEN}
    elif G_NUM.number_of_edges() > 0:
        return {"NUM": G_NUM}
    
    # All empty -> return empty graph so we still produce a row
    return {"SINGLE": nx.Graph()}

# ========================= Single-graph extractor =========================
def extract_features_single_graph(
    G: nx.Graph,
    groups: Dict[str, bool],
    *,
    ind4_exact_nmax: int,
    ind4_max_samples: int,
    ind5_sample_frac: float,
    seed: int,
    intra_workers: int,
) -> Dict[str, Any]:
    """Extract motif and spectral features from a single graph.
    
    This function computes various graph-theoretic features including:
    - Basic motifs (triangles, wedges, cycles, cliques)
    - Induced subgraph counts (4-node and 5-node graphlets)
    - Spectral features (Laplacian eigenvalues, adjacency spectrum)
    - Normalized counts (ratios relative to maximum possible)
    
    Args:
        G: NetworkX graph to analyze
        groups: Dictionary mapping feature group names to boolean flags
            - "motifs34": 3-motifs (triangles, wedges) and 4-motifs (4-cycles, 4-cliques)
            - "motifs5": 5-motifs (5-cycles, 5-cliques)
            - "induced4": Induced 4-node subgraphs (g_1^4 to g_6^4)
            - "induced5": Induced 5-node subgraphs (g_0^5 to g_20^5)
            - "cross_layer": Cross-layer motifs (for multi-layer graphs)
            - "spectral": Spectral features (Laplacian + adjacency spectrum)
        ind4_exact_nmax: Maximum nodes for exact 4-node induced subgraph counting
        ind4_max_samples: Maximum samples for approximate 4-node counting
        ind5_sample_frac: Fraction of 5-node subsets to sample (0.0 to 1.0)
        seed: Random seed for reproducible sampling
        intra_workers: Number of threads for parallel induced subgraph counting
    
    Returns:
        Dict[str, Any]: Dictionary of feature names to values
        
    Feature Categories:
        1. Basic Motifs:
           - Motif_triangles: Number of triangles (3-cliques)
           - Motif_wedges: Number of wedges (2-paths)
           - Motif_4_cycles: Number of 4-cycles
           - Motif_4_cliques: Number of 4-cliques (K4)
           - Motif_5_cycles: Number of 5-cycles
           - Motif_5_cliques: Number of 5-cliques (K5)
           
        2. Induced Subgraphs (4-node):
           - Motif_induced_g_1_4: Path4 (P4) - 4-node path
           - Motif_induced_g_2_4: Star4 (K1,3) - 4-node star
           - Motif_induced_g_3_4: Cycle4 (C4) - 4-node cycle
           - Motif_induced_g_4_4: TailedTriangle - triangle with tail
           - Motif_induced_g_5_4: Diamond - 4-node diamond
           - Motif_induced_g_6_4: Clique4 (K4) - 4-node clique
           
        3. Induced Subgraphs (5-node):
           - Motif_induced5_g_0_5 to g_20_5: All 21 canonical 5-node graphlets
           
        4. Normalized Counts:
           - *_per_Cn3: Ratio to C(n,3) = n!/(3!(n-3)!) - max possible 3-node subsets
           - *_per_Cn4: Ratio to C(n,4) = n!/(4!(n-4)!) - max possible 4-node subsets
           - *_per_Cn5: Ratio to C(n,5) = n!/(5!(n-5)!) - max possible 5-node subsets
           - *_per_max: Ratio to theoretical maximum for specific motif type
           
        5. Edge Statistics:
           - Motif_triangle_edge_incidence_*: Statistics on how many triangles each edge participates in
           - Motif_square_clustering_proxy: Tendency to form 4-cycles relative to 2-paths
           
        6. Spectral Features:
           - Spectral_algebraic_connectivity: Second smallest Laplacian eigenvalue (Fiedler value)
           - Spectral_spectral_gap: Difference between first two Laplacian eigenvalues
           - Spectral_lap_eig_0 to _9: First 10 Laplacian eigenvalues
           - Spectral_adjacency_*: Adjacency matrix spectrum features
    
    Notes:
        - Normalized features (e.g., *_per_Cn3) provide graph-size independent measures
        - Induced subgraph counting uses sampling for large graphs to avoid combinatorial explosion
        - All features return NaN for empty graphs or when computation fails
        - Cross-layer features are only computed for multi-layer graphs
    """

    # Initialize random number generator and feature dictionary
    rnd = random.Random(seed)
    enabled = lambda k: groups.get(k, False)
    feats: Dict[str, Any] = {}
    n = G.number_of_nodes(); m = G.number_of_edges()

    # ========================= BASIC MOTIFS (3/4-node) =========================
    # Precompute adjacency matrix and triangle count for efficiency
    if enabled("motifs34") or enabled("motifs5") or enabled("induced4") or enabled("induced5"):
        A = _adjacency_matrix(G)
        tri = count_triangles(G, A=A)
        
    if enabled("motifs34"):
        # Basic 3-motifs and 4-motifs
        feats["Motif_triangles"]   = tri  # Number of triangles (3-cliques)
        feats["Motif_wedges"]      = count_wedges(G, degs=[d for _, d in G.degree()], tri=tri)  # Number of 2-paths
        feats["Motif_4_cycles"]    = count_4_cycles(G, A=A)  # Number of 4-cycles
        feats["Motif_4_cliques"]   = count_4_cliques(G)  # Number of 4-cliques (K4)
        
        # Edge triangle incidence statistics - how many triangles each edge participates in
        m_mean, m_std = compute_edge_triangle_incidence_stats(G)
        feats["Motif_triangle_edge_incidence_mean"] = m_mean  # Average triangles per edge
        feats["Motif_triangle_edge_incidence_std"]  = m_std   # Std dev of triangles per edge
        feats["Motif_square_clustering_proxy"]      = compute_square_clustering_proxy(G, A=A)  # 4-cycle tendency
        
        # Additional edge statistics - edge embeddedness (common neighbors)
        emb_list = [len(set(G[u]) & set(G[v])) for u, v in G.edges()]  # Common neighbors for each edge
        feats["Motif_triangle_edge_incidence_median"] = _q(emb_list, 0.50)  # Median embeddedness
        feats["Motif_triangle_edge_incidence_q90"]    = _q(emb_list, 0.90)  # 90th percentile embeddedness
        feats["Motif_triangle_edge_frac_zero"]        = _frac(emb_list, lambda a: np.isclose(a, 0.0))  # Fraction with 0 common neighbors
        feats["Motif_triangle_edge_frac_ge2"]         = _frac(emb_list, lambda a: a >= 2)  # Fraction with ≥2 common neighbors

    # ========================= 5-NODE MOTIFS =========================
    if enabled("motifs5"):
        feats["Motif_5_cycles"]  = count_5_cycles(G)   # Number of 5-cycles
        feats["Motif_5_cliques"] = count_5_cliques(G)  # Number of 5-cliques (K5)

    # ========================= INDUCED 4-NODE SUBGRAPHS =========================
    # Enumerate all 4-node subsets and classify their induced subgraphs
    if enabled("induced4"):
        ind4 = _induced4_counts(G, max_samples=ind4_max_samples if n > ind4_exact_nmax else None, rng=rnd, n_jobs=intra_workers)
        total_4sets = math.comb(n, 4) if n >= 4 else 0  # C(n,4) = n!/(4!(n-4)!)
        for k, v in ind4.items():
            feats[f"Motif_induced_{k}"] = float(v)  # Raw counts for each 4-node graphlet type
        feats["Motif_induced_connected_per_4set"] = float(sum(ind4.values()) / total_4sets) if (total_4sets and total_4sets==total_4sets) else np.nan  # Fraction of 4-sets that are connected

    # ========================= INDUCED 5-NODE SUBGRAPHS =========================
    # Enumerate 5-node subsets with sampling for large graphs
    if enabled("induced5"):
        ind5, connected5 = _induced5_counts(G, sample_fraction=ind5_sample_frac, random_state=seed, n_jobs=intra_workers)
        total_5sets = math.comb(n, 5) if n >= 5 else 0  # C(n,5) = n!/(5!(n-5)!)
        for name, val in ind5.items():
            feats[f"Motif_induced5_{name}"] = float(val)  # Raw counts for each 5-node graphlet type
        feats["Motif_induced_connected_per_5set"] = (connected5 / total_5sets) if (total_5sets and total_5sets==total_5sets and connected5==connected5) else np.nan  # Fraction of 5-sets that are connected

    # ========================= NORMALIZED MOTIF COUNTS =========================
    # Compute ratios relative to maximum possible counts (graph-size independent measures)
    if enabled("motifs34") or enabled("motifs5") or enabled("induced4") or enabled("induced5"):
        Cn3 = math.comb(n, 3) if n >= 3 else 0  # C(n,3) = n!/(3!(n-3)!) - max possible 3-node subsets
        Cn4 = math.comb(n, 4) if n >= 4 else 0  # C(n,4) = n!/(4!(n-4)!) - max possible 4-node subsets
        Cn5 = math.comb(n, 5) if n >= 5 else 0  # C(n,5) = n!/(5!(n-5)!) - max possible 5-node subsets
        max_wedges = n * (n-1) * (n-2) // 6 if n >= 3 else 0  # Theoretical max wedges
        max_c5 = n * (n-1) * (n-2) * (n-3) * (n-4) // 120 if n >= 5 else 0  # Theoretical max 5-cycles
        
        if enabled("motifs34"):
            tri = feats.get("Motif_triangles", np.nan)
            c4  = feats.get("Motif_4_cycles", np.nan)
            k4  = feats.get("Motif_4_cliques", np.nan)
            w   = feats.get("Motif_wedges", np.nan)
            feats["Motif_triangles_per_Cn3"] = tri/Cn3 if Cn3 and Cn3==Cn3 else np.nan  # Triangles / max possible triangles
            feats["Motif_4_cycles_per_Cn4"]  = c4/Cn4  if Cn4 and Cn4==Cn4 else np.nan  # 4-cycles / max possible 4-cycles
            feats["Motif_4_cliques_per_Cn4"] = k4/Cn4  if Cn4 and Cn4==Cn4 else np.nan  # 4-cliques / max possible 4-cliques
            feats["Motif_wedges_per_max"]    = w/max_wedges if (max_wedges and max_wedges==max_wedges) else np.nan  # Wedges / theoretical max
        if enabled("induced4"):
            for key in ["g_1_4", "g_2_4", "g_3_4", "g_4_4", "g_5_4", "g_6_4"]:  # Path4, Star4, Cycle4, TailedTriangle, Diamond, Clique4
                raw = feats.get(f"Motif_induced_{key}", np.nan)
                feats[f"Motif_induced_{key}_per_Cn4"] = raw/Cn4 if Cn4 and Cn4==Cn4 else np.nan  # Graphlet count / max possible 4-sets
        if enabled("motifs5"):
            c5 = feats.get("Motif_5_cycles", np.nan)
            k5 = feats.get("Motif_5_cliques", np.nan)
            feats["Motif_5_cycles_per_Cn5"]  = c5/Cn5 if Cn5 and Cn5==Cn5 else np.nan  # 5-cycles / max possible 5-cycles
            feats["Motif_5_cliques_per_Cn5"] = k5/Cn5 if Cn5 and Cn5==Cn5 else np.nan  # 5-cliques / max possible 5-cliques
            feats["Motif_5_cycles_per_Kn"]   = c5/max_c5 if (max_c5 and max_c5==max_c5) else np.nan  # 5-cycles / theoretical max
        if enabled("induced5"):
            for name in _G5_NAMES:
                raw5 = feats.get(f"Motif_induced5_{name}", np.nan)
                feats[f"Motif_induced5_{name}_per_Cn5"] = raw5 / Cn5 if raw5 == raw5 else np.nan  # Graphlet count / max possible 5-sets

    # ========================= SPECTRAL FEATURES =========================
    # Laplacian eigenvalues and adjacency spectrum features
    if enabled("spectral"):
        # Laplacian eigenvalues and related features
        try:
            leigs = _laplacian_eigs(G)
            feats["Spectral_algebraic_connectivity"] = float(leigs[1])           if len(leigs) > 1 else np.nan  # Second smallest eigenvalue (Fiedler value)
            feats["Spectral_spectral_gap"]           = float(leigs[1]-leigs[0]) if len(leigs) > 1 else np.nan  # Difference between first two eigenvalues
            feats["Spectral_laplacian_mean"]         = float(np.mean(leigs)) if leigs.size else np.nan  # Mean of all eigenvalues
            feats["Spectral_laplacian_std"]          = float(np.std(leigs))  if leigs.size else np.nan  # Std dev of eigenvalues
            feats["Spectral_laplacian_skew"]         = float(pd.Series(leigs).skew()) if leigs.size > 2 else np.nan  # Skewness of eigenvalue distribution
            
            # First K Laplacian eigenvalues (K=10 by default)
            K = 10
            pad = np.full(K, np.nan); pad[:min(K, len(leigs))] = leigs[:K]
            feats.update({f"Spectral_lap_eig_{i}": float(pad[i]) for i in range(K)})  # Individual eigenvalues λ₀, λ₁, ..., λ₉
            
            # Kirchhoff index (sum of reciprocals of non-zero eigenvalues)
            nonzero = leigs[leigs > 1e-12]
            feats["Spectral_kirchhoff_index"] = float(n * np.sum(1.0 / nonzero)) if len(nonzero) >= 1 else np.nan  # Resistance-based connectivity measure
            
            # Laplacian heat traces at specific time points
            heat_times = [0.1, 0.5, 1.0, 2.0, 5.0]
            feats.update(_laplacian_heat_traces(leigs, heat_times))  # Heat kernel traces: Σ exp(-tλᵢ)
            
        except Exception:
            # Set all Laplacian features to NaN on error
            feats.update({
                "Spectral_algebraic_connectivity": np.nan,
                "Spectral_spectral_gap":           np.nan,
                "Spectral_laplacian_mean":         np.nan,
                "Spectral_laplacian_std":          np.nan,
                "Spectral_laplacian_skew":         np.nan,
                "Spectral_kirchhoff_index":         np.nan,
            })
            # Set first 10 eigenvalues to NaN
            feats.update({f"Spectral_lap_eig_{i}": np.nan for i in range(10)})
            # Set heat traces to NaN
            heat_times = [0.1, 0.5, 1.0, 2.0, 5.0]
            feats.update({f"Spectral_laplacian_heat_trace_t{t}": np.nan for t in heat_times})
        
        # Adjacency spectrum features
        try:
            A = _adjacency_matrix(G)
            energy, estrada, m2, m3, m4 = _adjacency_spectrum_features(A)
            feats["Spectral_adjacency_energy"]        = energy   # Sum of absolute eigenvalues
            feats["Spectral_adjacency_estrada_index"] = estrada  # Sum of exponentials of eigenvalues
            feats["Spectral_adjacency_moment_2"]      = m2       # Second moment of eigenvalues
            feats["Spectral_adjacency_moment_3"]      = m3       # Third moment of eigenvalues
            feats["Spectral_adjacency_moment_4"]      = m4       # Fourth moment of eigenvalues
        except Exception:
            feats.update({
                "Spectral_adjacency_energy":        np.nan,
                "Spectral_adjacency_estrada_index": np.nan,
                "Spectral_adjacency_moment_2":      np.nan,
                "Spectral_adjacency_moment_3":      np.nan,
                "Spectral_adjacency_moment_4":      np.nan,
            })

    return feats

# ========================= Row-level extractor =========================
def extract_features_row(
    row: pd.Series,
    groups: Dict[str, bool],
    *,
    ind4_exact_nmax: int,
    ind4_max_samples: int,
    ind5_sample_frac: float,
    seed: int,
    intra_workers: int,
) -> Dict[str, Any]:
    graphs = build_graphs_from_row(row)

    # Single-graph path (EDGES or empty)
    if "SINGLE" in graphs:
        feats = extract_features_single_graph(
            graphs["SINGLE"], groups,
            ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
            ind5_sample_frac=ind5_sample_frac,
            seed=seed, intra_workers=intra_workers,
        )
        if "COEFFICIENTS" in row: feats["COEFFICIENTS"] = row["COEFFICIENTS"]
        return feats

    # Layered path (DEN/NUM[/TOTAL]), possibly partial
    out: Dict[str, Any] = {}

    # Compute for whichever layers exist
    for tag in ("DEN","NUM","TOTAL"):
        if tag in graphs:
            feats_tag = extract_features_single_graph(
                graphs[tag], groups,
                ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
                ind5_sample_frac=ind5_sample_frac,
                seed=seed, intra_workers=intra_workers,
            )
            out.update({f"{k}_{tag}": v for k, v in feats_tag.items()})

    # Cross-layer metrics only if BOTH DEN and NUM exist
    if groups.get("cross_layer", False) and ("DEN" in graphs) and ("NUM" in graphs):
        G_DEN, G_NUM = graphs["DEN"], graphs["NUM"]
        G_TOTAL = nx.compose(G_DEN, G_NUM)
        n = G_TOTAL.number_of_nodes()
        e_den, e_num = G_DEN.number_of_edges(), G_NUM.number_of_edges()
        union_edges = set(G_DEN.edges()) | set(G_NUM.edges())
        
        # Cross-layer motif counts - motifs that span both layers
        cross_tri = count_cross_layer_motifs(G_DEN, G_NUM, "triangle")
        cross_c4  = count_cross_layer_motifs(G_DEN, G_NUM, "4cycle")
        cross_dia = count_cross_layer_motifs(G_DEN, G_NUM, "diamond")
        cross_k4  = count_cross_layer_motifs(G_DEN, G_NUM, "4clique")
        
        out.update({
            "Cross_layer_triangles": float(cross_tri),
            "Cross_layer_4cycles": float(cross_c4),
            "Cross_layer_diamonds": float(cross_dia),
            "Cross_layer_4cliques": float(cross_k4),
        })
        
        # Cross-layer ratios - fraction of motifs that are cross-layer
        tri_total = out.get("Motif_triangles_TOTAL", np.nan)
        c4_total  = out.get("Motif_4_cycles_TOTAL", np.nan)
        dia_total = out.get("Motif_induced_g_5_4_TOTAL", np.nan)  # Diamond
        k4_total  = out.get("Motif_induced_g_6_4_TOTAL", np.nan)  # Clique4
        
        out["Cross_layer_triangle_ratio"] = cross_tri / tri_total if tri_total and tri_total == tri_total else np.nan
        out["Cross_layer_4cycle_ratio"]  = cross_c4 / c4_total if c4_total and c4_total == c4_total else np.nan
        out["Cross_layer_diamond_ratio"]  = cross_dia / dia_total if dia_total and dia_total == dia_total else np.nan
        out["Cross_layer_4clique_ratio"]  = cross_k4 / k4_total if k4_total and k4_total == k4_total else np.nan

    if "COEFFICIENTS" in row: out["COEFFICIENTS"] = row["COEFFICIENTS"]
    return out

# ========================= Main processing =========================
def process_file(
    input_file: str,
    output_file: str,
    groups: List[str],
    *,
    batch_size: Optional[int] = None,
    batch_dir: Optional[str] = None,
    from_batch: int = 0,
    workers: int = 1,
    intra_workers: int = 0,
    ind4_exact_nmax: int = 1000,
    ind4_max_samples: int = 10000,
    ind5_sample_frac: float = 0.1,
    seed: int = 42,
    show_progress: bool = True,
) -> None:
    """Main processing function for motif feature extraction.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output parquet file
        groups: List of feature groups to extract
        batch_size: Process in batches of this size (optional)
        batch_dir: Directory to save batch results (optional)
        from_batch: Start from this batch number (for resuming)
        workers: Number of parallel workers for row processing
        intra_workers: Number of parallel workers within each row
        ind4_exact_nmax: Use exact counting for induced 4-node if n <= this
        ind4_max_samples: Max samples for induced 4-node when sampling
        ind5_sample_frac: Sample fraction for induced 5-node
        seed: Random seed for reproducibility
        show_progress: Whether to show progress bars
    """
    
    # Parse groups
    if "all" in groups:
        enabled_groups = {g: True for g in ALL_GROUPS}
    else:
        enabled_groups = {g: g in groups for g in ALL_GROUPS}
    
    print(f"Enabled groups: {[k for k, v in enabled_groups.items() if v]}")
    
    # Load data
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Process rows
    if batch_size and batch_dir:
        # Batched processing
        os.makedirs(batch_dir, exist_ok=True)
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(from_batch, total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx-1})")
            
            if _HAS_JOBLIB and workers > 1:
                if _HAS_TQDM_JOBLIB and show_progress:
                    with tqdm_joblib(tqdm(total=len(batch_df), desc=f"Batch {batch_idx + 1}")):
                        results = Parallel(n_jobs=workers, backend="loky")(
                            delayed(extract_features_row)(
                                row, enabled_groups,
                                ind4_exact_nmax=ind4_exact_nmax,
                                ind4_max_samples=ind4_max_samples,
                                ind5_sample_frac=ind5_sample_frac,
                                seed=seed,
                                intra_workers=intra_workers,
                            ) for _, row in batch_df.iterrows()
                        )
                else:
                    results = Parallel(n_jobs=workers, backend="loky")(
                        delayed(extract_features_row)(
                            row, enabled_groups,
                            ind4_exact_nmax=ind4_exact_nmax,
                            ind4_max_samples=ind4_max_samples,
                            ind5_sample_frac=ind5_sample_frac,
                            seed=seed,
                            intra_workers=intra_workers,
                        ) for _, row in batch_df.iterrows()
                    )
            else:
                results = []
                for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx + 1}", disable=not show_progress):
                    result = extract_features_row(
                        row, enabled_groups,
                        ind4_exact_nmax=ind4_exact_nmax,
                        ind4_max_samples=ind4_max_samples,
                        ind5_sample_frac=ind5_sample_frac,
                        seed=seed,
                        intra_workers=intra_workers,
                    )
                    results.append(result)
            
            # Save batch - detect format from main output file
            output_format = _detect_output_format(output_file)
            batch_ext = '.csv' if output_format == 'csv' else '.parquet'
            batch_output = os.path.join(batch_dir, f"batch_{batch_idx:04d}{batch_ext}")
            _save_dataframe(pd.DataFrame(results), batch_output, output_format)
            print(f"Saved batch to {batch_output}")
        
        print(f"Batched processing complete. Batches saved to {batch_dir}")
        
    else:
        # Single file processing
        if _HAS_JOBLIB and workers > 1:
            if _HAS_TQDM_JOBLIB and show_progress:
                with tqdm_joblib(tqdm(total=len(df), desc="Processing")):
                    results = Parallel(n_jobs=workers, backend="loky")(
                        delayed(extract_features_row)(
                            row, enabled_groups,
                            ind4_exact_nmax=ind4_exact_nmax,
                            ind4_max_samples=ind4_max_samples,
                            ind5_sample_frac=ind5_sample_frac,
                            seed=seed,
                            intra_workers=intra_workers,
                        ) for _, row in df.iterrows()
                    )
            else:
                results = Parallel(n_jobs=workers, backend="loky")(
                    delayed(extract_features_row)(
                        row, enabled_groups,
                        ind4_exact_nmax=ind4_exact_nmax,
                        ind4_max_samples=ind4_max_samples,
                        ind5_sample_frac=ind5_sample_frac,
                        seed=seed,
                        intra_workers=intra_workers,
                    ) for _, row in df.iterrows()
                )
        else:
            results = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not show_progress):
                result = extract_features_row(
                    row, enabled_groups,
                    ind4_exact_nmax=ind4_exact_nmax,
                    ind4_max_samples=ind4_max_samples,
                    ind5_sample_frac=ind5_sample_frac,
                    seed=seed,
                    intra_workers=intra_workers,
                )
                results.append(result)
        
        # Save results
        result_df = pd.DataFrame(results)
        _save_dataframe(result_df, output_file)
        print(f"Saved {len(result_df)} rows to {output_file}")
        
        # Create manifest
        manifest = {
            "input_file": input_file,
            "output_file": output_file,
            "output_format": _detect_output_format(output_file),
            "groups": groups,
            "num_rows": len(result_df),
            "features": list(result_df.columns),
            "num_features": len(result_df.columns),
        }
        manifest_file = output_file.replace('.parquet', '_manifest.json').replace('.csv', '_manifest.json')
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_file}")

# ========================= CLI =========================
def main():
    parser = argparse.ArgumentParser(description="Extract motif-based features from graphs")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output file (CSV or Parquet format)")
    parser.add_argument("--groups", default="all", 
                       help="Comma-separated list of groups: motifs34,motifs5,induced4,induced5,cross_layer,spectral (default: all)")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, help="Process in batches of this size")
    parser.add_argument("--batch-dir", help="Directory to save batches")
    parser.add_argument("--from-batch", type=int, default=0, help="Start from this batch number")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--intra-workers", type=int, default=0, help="Number of intra-row parallel workers")
    
    # Feature-specific options
    parser.add_argument("--ind4-exact-nmax", type=int, default=1000, 
                       help="Use exact counting for induced 4-node if n <= this (default: 1000)")
    parser.add_argument("--ind4-max-samples", type=int, default=10000, 
                       help="Max samples for induced 4-node when sampling (default: 10000)")
    parser.add_argument("--ind5-sample-frac", type=float, default=0.1, 
                       help="Sample fraction for induced 5-node (default: 0.1)")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    
    args = parser.parse_args()
    
    # Parse groups
    groups = [g.strip() for g in args.groups.split(",")]
    
    # Validate groups
    valid_groups = set(ALL_GROUPS) | {"all"}
    invalid_groups = set(groups) - valid_groups
    if invalid_groups:
        print(f"Warning: Invalid groups ignored: {invalid_groups}")
        groups = [g for g in groups if g in valid_groups]
    
    if not groups:
        print("Error: No valid groups specified")
        return 1
    
    # Validate output format
    output_format = _detect_output_format(args.output)
    if output_format not in ['csv', 'parquet']:
        print(f"Error: Unsupported output format. Use .csv or .parquet extension")
        return 1
    
    print(f"Output format: {output_format}")
    
    # Process file
    try:
        process_file(
            input_file=args.input,
            output_file=args.output,
            groups=groups,
            batch_size=args.batch_size,
            batch_dir=args.batch_dir,
            from_batch=args.from_batch,
            workers=args.workers,
            intra_workers=args.intra_workers,
            ind4_exact_nmax=args.ind4_exact_nmax,
            ind4_max_samples=args.ind4_max_samples,
            ind5_sample_frac=args.ind5_sample_frac,
            seed=args.seed,
            show_progress=not args.no_progress,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
