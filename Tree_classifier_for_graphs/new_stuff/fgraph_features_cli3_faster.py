#!/usr/bin/env python3
# fgraph_features_selected_cli.py
# -------------------------------------------------------------
# Graph feature extractor restricted to a specific feature set.
# It merges logic from:
#   - fgraph_features_cli.py (full feature extractor)
#   - motif_features_cli.py (motif + spectral variants)
#
# It outputs exactly the following columns (in this order):
#   DESIRED_COLUMNS (see definition below)
#
# Supports:
#   - Single-layer graphs via EDGES
#   - Two-layer graphs via DEN_EDGES + NUM_EDGES
#   - Optional COEFFICIENTS and Unnamed: 0 columns (carried through)
#   - Batching with resume
#   - Manifest JSON
#   - Row-level parallelism + intra-row parallelism
# -------------------------------------------------------------

import os, sys, argparse, ast, warnings, math, random, json
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

# Suppress all UserWarnings (including networkx hash warnings)
warnings.filterwarnings("ignore", category=UserWarning)
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

# s3

import subprocess

def s3_upload(local_path: str, s3_uri: str, profile: str = None):
    # Ensure trailing slash behavior is sane: we upload the file into that prefix
    s3_uri = s3_uri.rstrip("/") + "/"
    cmd = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["s3", "cp", local_path, s3_uri]
    subprocess.run(cmd, check=True)

import io

def write_frame_to_s3_stream(df: pd.DataFrame, s3_uri: str, filename: str, fmt: str):
    """
    Streams DataFrame directly to S3 using aws s3 cp - s3://...
    """
    s3_path = s3_uri.rstrip("/") + "/" + filename

    if fmt == "csv":
        cmd = ["aws", "s3", "cp", "-", s3_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        df.to_csv(proc.stdin, index=False)
        proc.stdin.close()
        proc.wait()
    else:
        # parquet requires a bytes buffer
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        cmd = ["aws", "s3", "cp", "-", s3_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        proc.stdin.write(buf.read())
        proc.stdin.close()
        proc.wait()



# ========================= Desired feature set =========================
DESIRED_COLUMNS = [
 'Adjacency_energy',
 'Adjacency_energy_over_fro',
 'Adjacency_energy_per_node',
 'Adjacency_estrada_index',
 'Adjacency_estrada_per_node',
 'Adjacency_moment_2',
 'Adjacency_moment_2_over_avgdeg',
 'Adjacency_moment_3',
 'Adjacency_moment_3_over_avgdeg3',
 'Adjacency_moment_4',
 'Adjacency_moment_4_over_avgdeg4',
 'Assortativity_degree',
 'Basic_avg_degree',
 'Basic_avg_degree_norm',
 'Basic_degree_entropy',
 'Basic_degree_entropy_norm',
 'Basic_degree_skew',
 'Basic_degree_std',
 'Basic_density',
 'Basic_edge_to_node_ratio',
 'Basic_max_degree',
 'Basic_min_degree',
 'Basic_num_edges',
 'Basic_num_nodes',
 'COEFFICIENTS',
 'Centrality_betweenness_max',
 'Centrality_betweenness_mean',
 'Centrality_betweenness_skew',
 'Centrality_betweenness_std',
 'Centrality_closeness_max',
 'Centrality_closeness_max_norm',
 'Centrality_closeness_mean',
 'Centrality_closeness_mean_norm',
 'Centrality_closeness_skew',
 'Centrality_closeness_std',
 'Centrality_eigenvector_max',
 'Centrality_eigenvector_mean',
 'Centrality_eigenvector_skew',
 'Centrality_eigenvector_std',
 'Clustering_frac_one',
 'Clustering_frac_zero',
 'Clustering_mean',
 'Clustering_q10',
 'Clustering_q50',
 'Clustering_q90',
 'Comm_count',
 'Comm_internal_edge_frac',
 'Comm_modularity',
 'Comm_size_gini',
 'Comm_size_max',
 'Connectivity_avg_shortest_path_length',
 'Connectivity_diameter',
 'Connectivity_diameter_norm',
 'Connectivity_is_connected',
 'Connectivity_num_components',
 'Connectivity_num_components_per_node',
 'Connectivity_radius',
 'Connectivity_radius_norm',
 'Connectivity_wiener_index',
 'Core_core_index_mean',
 'Core_max_core_index',
 'Cycle_num_cycles_len_5',
 'Cycle_num_cycles_len_6',
 'Degree_gini',
 'Ecc_mean',
 'Ecc_q90',
 'Eff_diameter_p90',
 'Kirchhoff_index',
 'Motif_4_cliques',
 'Motif_4_cliques_per_Cn4',
 'Motif_4_cycles',
 'Motif_4_cycles_per_Cn4',
 'Motif_induced_C4',
 'Motif_induced_C4_per_Cn4',
 'Motif_induced_Diamond',
 'Motif_induced_Diamond_per_Cn4',
 'Motif_induced_K1_3',
 'Motif_induced_K1_3_per_Cn4',
 'Motif_induced_K4',
 'Motif_induced_K4_per_Cn4',
 'Motif_induced_P4',
 'Motif_induced_P4_per_Cn4',
 'Motif_induced_TailedTriangle',
 'Motif_induced_TailedTriangle_per_Cn4',
 'Motif_induced_connected_per_4set',
 'Motif_square_clustering_proxy',
 'Motif_triangle_edge_frac_ge2',
 'Motif_triangle_edge_frac_zero',
 'Motif_triangle_edge_incidence_mean',
 'Motif_triangle_edge_incidence_median',
 'Motif_triangle_edge_incidence_q90',
 'Motif_triangle_edge_incidence_std',
 'Motif_triangles',
 'Motif_triangles_per_Cn3',
 'Motif_wedges',
 'Motif_wedges_per_max',
 'NetLSD_mean',
 'NetLSD_q10',
 'NetLSD_q90',
 'NetLSD_std',
 'Planarity_face_size_max',
 'Planarity_face_size_mean',
 'Planarity_face_size_mean_norm',
 'Planarity_num_faces',
 'Planarity_num_faces_over_upperbound',
 'Robust_articulation_points',
 'Robust_articulation_points_per_node',
 'Robust_bridge_count',
 'Robust_bridge_count_per_edge',
 'Spectral_algebraic_connectivity',
 'Spectral_algebraic_connectivity_over_avgdeg',
 'Spectral_lap_eig_0',
 'Spectral_lap_eig_1',
 'Spectral_lap_eig_2',
 'Spectral_lap_eig_3',
 'Spectral_lap_eig_4',
 'Spectral_lap_eig_5',
 'Spectral_lap_eig_6',
 'Spectral_lap_eig_7',
 'Spectral_lap_eig_8',
 'Spectral_lap_eig_9',
 'Spectral_laplacian_heat_trace_t0.1',
 'Spectral_laplacian_heat_trace_t0.1_per_node',
 'Spectral_laplacian_heat_trace_t1.0',
 'Spectral_laplacian_heat_trace_t1.0_per_node',
 'Spectral_laplacian_heat_trace_t5.0',
 'Spectral_laplacian_heat_trace_t5.0_per_node',
 'Spectral_laplacian_mean',
 'Spectral_laplacian_skew',
 'Spectral_laplacian_std',
 'Spectral_spectral_gap',
 'Spectral_spectral_gap_rel',
 'Symmetry_aut_size_log_over_log_nfact',
 'Symmetry_automorphism_group_order',
 'Symmetry_num_orbits',
 'Symmetry_num_orbits_per_node',
 'Symmetry_orbit_size_max',
 'Symmetry_orbit_size_max_per_node',
 'TDA_Betti0_at_q25',
 'TDA_Betti0_at_q25_per_node',
 'TDA_Betti0_at_q50',
 'TDA_Betti0_at_q50_per_node',
 'TDA_Betti0_at_q75',
 'TDA_Betti0_at_q75_per_node',
 'TDA_Betti1_at_q25',
 'TDA_Betti1_at_q25_per_node',
 'TDA_Betti1_at_q50',
 'TDA_Betti1_at_q50_per_node',
 'TDA_Betti1_at_q75',
 'TDA_Betti1_at_q75_per_node',
 'TDA_H0_count',
 'TDA_H0_count_per_node',
 'TDA_H0_max_persistence',
 'TDA_H0_max_persistence_over_diam',
 'TDA_H0_mean_birth',
 'TDA_H0_mean_birth_over_diam',
 'TDA_H0_mean_death',
 'TDA_H0_mean_death_over_diam',
 'TDA_H0_mean_persistence',
 'TDA_H0_mean_persistence_over_diam',
 'TDA_H0_persistence_entropy',
 'TDA_H0_total_persistence',
 'TDA_H0_total_persistence_over_diam',
 'TDA_H1_count',
 'TDA_H1_count_per_node',
 'TDA_H1_max_persistence',
 'TDA_H1_max_persistence_over_diam',
 'TDA_H1_mean_birth',
 'TDA_H1_mean_birth_over_diam',
 'TDA_H1_mean_death',
 'TDA_H1_mean_death_over_diam',
 'TDA_H1_mean_persistence',
 'TDA_H1_mean_persistence_over_diam',
 'TDA_H1_persistence_entropy',
 'TDA_H1_total_persistence',
 'TDA_H1_total_persistence_over_diam',
 'Unnamed: 0',
 'Wiener_mean_distance',
 'log_Adjacency_estrada_per_node',
 'Motif_5_cliques',
 'Motif_5_cliques_per_Cn5',
 'Motif_5_cycles',
 'Motif_5_cycles_per_Cn5',
 'Motif_5_cycles_per_Kn',
 'Motif_induced5_g_0_5',
 'Motif_induced5_g_0_5_per_Cn5',
 'Motif_induced5_g_10_5',
 'Motif_induced5_g_10_5_per_Cn5',
 'Motif_induced5_g_11_5',
 'Motif_induced5_g_11_5_per_Cn5',
 'Motif_induced5_g_12_5',
 'Motif_induced5_g_12_5_per_Cn5',
 'Motif_induced5_g_13_5',
 'Motif_induced5_g_13_5_per_Cn5',
 'Motif_induced5_g_14_5',
 'Motif_induced5_g_14_5_per_Cn5',
 'Motif_induced5_g_15_5',
 'Motif_induced5_g_15_5_per_Cn5',
 'Motif_induced5_g_16_5',
 'Motif_induced5_g_16_5_per_Cn5',
 'Motif_induced5_g_17_5',
 'Motif_induced5_g_17_5_per_Cn5',
 'Motif_induced5_g_18_5',
 'Motif_induced5_g_18_5_per_Cn5',
 'Motif_induced5_g_1_5',
 'Motif_induced5_g_1_5_per_Cn5',
 'Motif_induced5_g_20_5',
 'Motif_induced5_g_20_5_per_Cn5',
 'Motif_induced5_g_2_5',
 'Motif_induced5_g_2_5_per_Cn5',
 'Motif_induced5_g_3_5',
 'Motif_induced5_g_3_5_per_Cn5',
 'Motif_induced5_g_4_5',
 'Motif_induced5_g_4_5_per_Cn5',
 'Motif_induced5_g_5_5',
 'Motif_induced5_g_5_5_per_Cn5',
 'Motif_induced5_g_6_5',
 'Motif_induced5_g_6_5_per_Cn5',
 'Motif_induced5_g_7_5',
 'Motif_induced5_g_7_5_per_Cn5',
 'Motif_induced5_g_8_5',
 'Motif_induced5_g_8_5_per_Cn5',
 'Motif_induced5_g_9_5',
 'Motif_induced5_g_9_5_per_Cn5',
 'Motif_induced_connected_per_5set',
 'Motif_induced_g_1_4',
 'Motif_induced_g_1_4_per_Cn4',
 'Motif_induced_g_2_4',
 'Motif_induced_g_2_4_per_Cn4',
 'Motif_induced_g_3_4',
 'Motif_induced_g_3_4_per_Cn4',
 'Motif_induced_g_4_4',
 'Motif_induced_g_4_4_per_Cn4',
 'Motif_induced_g_5_4',
 'Motif_induced_g_5_4_per_Cn4',
 'Motif_induced_g_6_4',
 'Motif_induced_g_6_4_per_Cn4',
 'Spectral_adjacency_energy',
 'Spectral_adjacency_estrada_index',
 'Spectral_adjacency_moment_2',
 'Spectral_adjacency_moment_3',
 'Spectral_adjacency_moment_4',
 'Spectral_kirchhoff_index',
 'Spectral_laplacian_heat_trace_t0.5',
 'Spectral_laplacian_heat_trace_t2.0',
]

# ========================= Feature groups =========================
ALL_GROUPS = [
    "basic","connectivity","centrality","core","robustness","cycles",
    "spectral_laplacian","spectral_adjacency","netlsd","planarity","symmetry",
    "community","motifs34","motifs5","induced4","induced5","tda","cross_layer",
]
DEFAULT_ENABLED_GROUPS = {k: True for k in ALL_GROUPS}

def parse_groups(arg_groups: Optional[str]) -> Dict[str, bool]:
    if (arg_groups is None) or (arg_groups.strip().lower() == "all"):
        return DEFAULT_ENABLED_GROUPS.copy()
    base = {k: False for k in ALL_GROUPS}
    req = [s.strip().lower() for s in arg_groups.split(",") if s.strip()]
    for k in req:
        if k in base: base[k] = True
    return base

# ========================= tqdm helper =========================
def _tqdm(it=None, **kwargs):
    # Force tqdm to write to stderr and ensure it's visible
    kwargs.setdefault('file', sys.stderr)
    kwargs.setdefault('dynamic_ncols', True)
    kwargs.setdefault('mininterval', 0.1)
    kwargs.setdefault('smoothing', 0.1)
    kwargs.setdefault('leave', False)
    # Force disable=False to ensure progress bar shows even if not a TTY
    kwargs.setdefault('disable', False)
    return tqdm(it, **kwargs)

# ========================= Helpers =========================
def shannon_entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    return np.nan if tot == 0 else -sum((c / tot) * log2(c / tot) for c in counter.values())

def try_or_nan(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return np.nan

def safe_comb(n, k):
    if n is None:
        return np.nan
    try:
        n = int(n)
        if n >= k:
            return math.comb(n, k)
        return np.nan
    except Exception:
        return np.nan

def gini(x):
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2*np.sum(cum)/cum[-1]) / n)

def _q(vals, q):
    if not vals:
        return np.nan
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else np.nan

def _frac(arr, pred):
    if not arr:
        return np.nan
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.mean(pred(a)))

def extract_faces(embedding: nx.PlanarEmbedding):
    seen = set(); faces = []
    for u in embedding:
        for v in embedding[u]:
            if (u, v) in seen:
                continue
            face = embedding.traverse_face(u, v)
            faces.append(face)
            seen.update((face[i], face[(i+1)%len(face)]) for i in range(len(face)))
    return faces

def compute_node_orbits(graph):
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph)
    orbit_map = defaultdict(set)
    for iso in matcher.isomorphisms_iter():
        for u, v in iso.items():
            orbit_map[u].add(v)
    seen = set(); orbits = []
    for group in orbit_map.values():
        g_frozen = frozenset(group)
        if g_frozen not in seen:
            seen.add(g_frozen)
            orbits.append(group)
    return orbits

def _safe_degree_assortativity(G: nx.Graph) -> float:
    try:
        if G.number_of_edges() == 0:
            return np.nan
        degs = [d for _, d in G.degree()]
        if len(set(degs)) <= 1:
            return np.nan
        return float(nx.degree_pearson_correlation_coefficient(G))
    except Exception:
        return np.nan

# ========================= Motif helpers =========================
def _adjacency_matrix(G):
    return nx.to_numpy_array(G, dtype=float, nodelist=list(G.nodes()))

def _triangle_count(G, A=None):
    if A is None:
        A = _adjacency_matrix(G)
    A3 = A @ A @ A
    return int(round(np.trace(A3) / 6.0))

def _wedge_count(G, degs=None, tri=None):
    if degs is None:
        degs = [d for _, d in G.degree()]
    if tri is None:
        tri = _triangle_count(G)
    s = sum(d*(d-1)//2 for d in degs)
    return int(s - 3*tri)

def _four_cycle_count(G, A=None):
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

def _four_clique_count(G):
    cnt = 0
    for clq in nx.find_cliques(G):
        k = len(clq)
        if k >= 4:
            cnt += math.comb(k, 4)
    return int(cnt)

def _edge_triangle_incidence_stats(G):
    out = []
    for u, v in G.edges():
        out.append(len(set(G[u]) & set(G[v])))
    if not out:
        return 0.0, 0.0
    arr = np.array(out, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))

def _square_clustering_proxy(G, A=None):
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
    c4 = _four_cycle_count(G, A=A)
    if denom == 0:
        return np.nan
    return float(4.0 * c4 / denom)

# ---- induced 4-node (original classification) ----
def _classify_induced4(G_sub):
    if not nx.is_connected(G_sub): return None
    m = G_sub.number_of_edges()
    degs = sorted([d for _, d in G_sub.degree()])
    if m == 3:
        if degs == [1,1,2,2]: return "P4"
        if degs == [1,1,1,3]: return "K1_3"
    elif m == 4:
        if degs == [2,2,2,2]: return "C4"
        if degs == [1,2,2,3]: return "TailedTriangle"
    elif m == 5:
        if degs == [2,2,3,3]: return "Diamond"
    elif m == 6:
        if degs == [3,3,3,3]: return "K4"
    return None

### new for faster
# 6 edges among 4 nodes, fixed order for bitmask
_EDGES4 = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

def _popcount6(x: int) -> int:
    # Python 3.8+: int.bit_count() is very fast
    return x.bit_count()

def _mask4_class(mask: int):
    """
    Return one of: 'P4','K1_3','C4','TailedTriangle','Diamond','K4' or None.
    Works on the 6-bit edge mask in _EDGES4 order.
    """
    m = _popcount6(mask)
    if m < 3:
        return None
    if m == 6:
        return "K4"

    # degrees from mask (4 nodes)
    deg = [0,0,0,0]
    for bit, (i,j) in enumerate(_EDGES4):
        if (mask >> bit) & 1:
            deg[i] += 1; deg[j] += 1
    degs = sorted(deg)

    # Connectedness on 4 nodes can be checked cheaply via masks:
    # Any connected 4-node graph has at least one spanning tree (3 edges) and no isolated nodes.
    # We can reject isolated nodes quickly:
    if degs[0] == 0:
        return None

    if m == 3:
        if degs == [1,1,2,2]:
            return "P4"
        if degs == [1,1,1,3]:
            return "K1_3"
        return None

    if m == 4:
        if degs == [2,2,2,2]:
            return "C4"
        if degs == [1,2,2,3]:
            return "TailedTriangle"
        return None

    if m == 5:
        if degs == [2,2,3,3]:
            return "Diamond"
        return None

    return None

# Build LUT once
_IND4_LUT = [None] * 64
for _mask in range(64):
    _IND4_LUT[_mask] = _mask4_class(_mask)


def _nbr_masks_from_graph(G: nx.Graph, nodes: list):
    """
    Build neighbor bitmask per node index for nodes list.
    Supports up to 64 nodes (you are ~20).
    """
    idx = {u:i for i,u in enumerate(nodes)}
    n = len(nodes)
    masks = [0] * n
    for u, v in G.edges():
        iu = idx.get(u); iv = idx.get(v)
        if iu is None or iv is None or iu == iv:
            continue
        masks[iu] |= (1 << iv)
        masks[iv] |= (1 << iu)
    return masks

def _induced4_counts_bitmask(G: nx.Graph, *, max_samples=None, rng=None):
    counts0 = {"K1_3": 0, "P4": 0, "C4": 0, "TailedTriangle": 0, "Diamond": 0, "K4": 0}
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 4:
        return counts0

    # Precompute neighbor masks once
    nbr = _nbr_masks_from_graph(G, nodes)

    # Choose 4-sets: exact or sampled (same behavior as your current code)
    if max_samples is None:
        combos = combinations(range(n), 4)
    else:
        rng = rng or random.Random(42)
        total = math.comb(n, 4)
        k = min(max_samples, total)
        seen = set()
        while len(seen) < k:
            S = tuple(sorted(rng.sample(range(n), 4)))
            seen.add(S)
        combos = seen

    for a, b, c, d in combos:
        # build 6-bit mask in _EDGES4 order, using bit checks
        mask = 0
        # (0,1)
        if (nbr[a] >> b) & 1: mask |= 1 << 0
        # (0,2)
        if (nbr[a] >> c) & 1: mask |= 1 << 1
        # (0,3)
        if (nbr[a] >> d) & 1: mask |= 1 << 2
        # (1,2)
        if (nbr[b] >> c) & 1: mask |= 1 << 3
        # (1,3)
        if (nbr[b] >> d) & 1: mask |= 1 << 4
        # (2,3)
        if (nbr[c] >> d) & 1: mask |= 1 << 5

        lab = _IND4_LUT[mask]
        if lab is not None:
            counts0[lab] += 1

    return counts0


# ========================= Spectral helpers =========================
def _laplacian_eigs_old(G):
    if G.number_of_nodes() == 0:
        return np.array([])
    L = nx.laplacian_matrix(G).todense()
    return np.sort(np.linalg.eigvalsh(L))

def _laplacian_eigs(G):
    n = G.number_of_nodes()
    if n == 0:
        return np.array([])

    # Keep node order deterministic
    nodes = list(G.nodes())

    # Dense adjacency (fast for n~20)
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)

    # L = D - A, but avoid allocating D explicitly
    deg = A.sum(axis=1)
    L = -A
    np.fill_diagonal(L, deg)

    # Laplacian eigenvalues are real for undirected graphs
    return np.sort(np.linalg.eigvalsh(L))

def _laplacian_heat_traces(leigs, ts: Iterable[float]):
    traces = {}
    if leigs is None or len(leigs) == 0 or np.any(np.isnan(leigs)):
        for t in ts: traces[f"Spectral_laplacian_heat_trace_t{t}"] = np.nan
        return traces
    for t in ts:
        traces[f"Spectral_laplacian_heat_trace_t{t}"] = float(np.sum(np.exp(-t * leigs)))
    return traces

def _adjacency_spectrum_features(A):
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

# ========================= TDA helpers =========================
try:
    from ripser import ripser
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False

def _tda_vr_h_summary_from_spdm(G):
    out = {
        "TDA_H0_count": np.nan, "TDA_H0_total_persistence": np.nan,
        "TDA_H0_mean_persistence": np.nan, "TDA_H0_max_persistence": np.nan,
        "TDA_H0_persistence_entropy": np.nan,
        "TDA_H0_mean_birth": np.nan, "TDA_H0_mean_death": np.nan,
        "TDA_H1_count": np.nan, "TDA_H1_total_persistence": np.nan,
        "TDA_H1_mean_persistence": np.nan, "TDA_H1_max_persistence": np.nan,
        "TDA_H1_persistence_entropy": np.nan,
        "TDA_H1_mean_birth": np.nan, "TDA_H1_mean_death": np.nan,
        "TDA_Betti0_at_q25": np.nan, "TDA_Betti0_at_q50": np.nan, "TDA_Betti0_at_q75": np.nan,
        "TDA_Betti1_at_q25": np.nan, "TDA_Betti1_at_q50": np.nan, "TDA_Betti1_at_q75": np.nan,
    }
    if not _HAS_RIPSER:
        return out
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    if len(comps) == 0:
        return out

    dists = []; comp_dist = []
    for H in comps:
        nodes = list(H.nodes())
        if H.number_of_nodes() == 1:
            comp_dist.append(None); continue
        D = np.array(nx.floyd_warshall_numpy(H, nodelist=nodes), dtype=float)
        vals = D[~np.eye(D.shape[0], dtype=bool)]
        if vals.size > 0:
            dists.extend(vals.tolist())
        comp_dist.append((D, nodes))
    if len(dists) == 0:
        out.update({
            "TDA_H0_count": 0.0, "TDA_H0_total_persistence": 0.0,
            "TDA_H0_mean_persistence": 0.0, "TDA_H0_max_persistence": 0.0,
            "TDA_H0_persistence_entropy": 0.0,
            "TDA_H0_mean_birth": 0.0, "TDA_H0_mean_death": 0.0,
            "TDA_H1_count": 0.0, "TDA_H1_total_persistence": 0.0,
            "TDA_H1_mean_persistence": 0.0, "TDA_H1_max_persistence": 0.0,
            "TDA_H1_persistence_entropy": 0.0,
            "TDA_H1_mean_birth": 0.0, "TDA_H1_mean_death": 0.0,
            "TDA_Betti0_at_q25": 1.0, "TDA_Betti0_at_q50": 1.0, "TDA_Betti0_at_q75": 1.0,
            "TDA_Betti1_at_q25": 0.0, "TDA_Betti1_at_q50": 0.0, "TDA_Betti1_at_q75": 0.0,
        })
        return out
    q25, q50, q75 = np.quantile(np.array(dists), [0.25, 0.50, 0.75])

    def _agg_ph(dgm):
        finite = dgm[np.isfinite(dgm[:,1])] if dgm.size else np.empty((0,2))
        if finite.size == 0:
            return dict(count=0.0, total=0.0, mean=0.0, max_=0.0, ent=0.0,
                        mean_birth=0.0, mean_death=0.0)
        pers = finite[:,1] - finite[:,0]
        total = float(np.sum(pers))
        meanp = float(np.mean(pers))
        maxp  = float(np.max(pers))
        probs = pers / (total if total > 0 else 1.0)
        ent = -float(np.sum([p*np.log(p) for p in probs if p > 0]))
        return dict(count=float(finite.shape[0]), total=total, mean=meanp, max_=maxp, ent=ent,
                    mean_birth=float(np.mean(finite[:,0])), mean_death=float(np.mean(finite[:,1])))

    def _betti_at(dgm, t):
        if dgm.size == 0:
            return 0.0
        birth = dgm[:,0]; death = dgm[:,1]
        return float(np.sum((birth <= t) & (t < death)))

    H0_list, H1_list = [], []
    betti0_q = [0.0,0.0,0.0]; betti1_q = [0.0,0.0,0.0]
    for item in comp_dist:
        if item is None:
            betti0_q = [b+1.0 for b in betti0_q]; continue
        D, _ = item
        res = ripser(D, distance_matrix=True, maxdim=1)
        dgms = res.get("dgms", [])
        H0 = dgms[0] if len(dgms) > 0 else np.empty((0,2))
        H1 = dgms[1] if len(dgms) > 1 else np.empty((0,2))
        H0_list.append(_agg_ph(H0)); H1_list.append(_agg_ph(H1))
        for idx, t in enumerate((q25, q50, q75)):
            betti0_q[idx] += _betti_at(H0, t)
            betti1_q[idx] += _betti_at(H1, t)

    def _combine(stats_list):
        if not stats_list:
            return dict(count=0.0, total=0.0, mean=0.0, max_=0.0, ent=0.0, mean_birth=0.0, mean_death=0.0)
        count = float(sum(s["count"] for s in stats_list))
        total = float(sum(s["total"] for s in stats_list))
        mean  = float(np.mean([s["mean"] for s in stats_list]))
        max_  = float(max([s["max_"] for s in stats_list]))
        ent   = float(np.mean([s["ent"] for s in stats_list]))
        mean_birth = float(np.mean([s["mean_birth"] for s in stats_list]))
        mean_death = float(np.mean([s["mean_death"] for s in stats_list]))
        return dict(count=count, total=total, mean=mean, max_=max_, ent=ent,
                    mean_birth=mean_birth, mean_death=mean_death)

    H0c = _combine(H0_list); H1c = _combine(H1_list)
    out.update({
        "TDA_H0_count": H0c["count"], "TDA_H0_total_persistence": H0c["total"],
        "TDA_H0_mean_persistence": H0c["mean"], "TDA_H0_max_persistence": H0c["max_"],
        "TDA_H0_persistence_entropy": H0c["ent"], "TDA_H0_mean_birth": H0c["mean_birth"], "TDA_H0_mean_death": H0c["mean_death"],
        "TDA_H1_count": H1c["count"], "TDA_H1_total_persistence": H1c["total"],
        "TDA_H1_mean_persistence": H1c["mean"], "TDA_H1_max_persistence": H1c["max_"],
        "TDA_H1_persistence_entropy": H1c["ent"], "TDA_H1_mean_birth": H1c["mean_birth"], "TDA_H1_mean_death": H1c["mean_death"],
        "TDA_Betti0_at_q25": betti0_q[0], "TDA_Betti0_at_q50": betti0_q[1], "TDA_Betti0_at_q75": betti0_q[2],
        "TDA_Betti1_at_q25": betti1_q[0], "TDA_Betti1_at_q50": betti1_q[1], "TDA_Betti1_at_q75": betti1_q[2],
    })
    return out
# ========================= 3/4-motifs - for faster computations =========================
def _triangles_from_A(A: np.ndarray) -> int:
    # triangles = trace(A^3)/6 for simple undirected 0/1 adjacency
    A3 = A @ A @ A
    return int(round(np.trace(A3) / 6.0))

def _four_cycles_and_denom_from_A2(A2: np.ndarray) -> (int, int):
    """
    Matches your current _four_cycle_count + denom logic:
      total_pairs = sum_{i<j} C(A2[i,j], 2)
      c4 = total_pairs // 2
      denom = total_pairs
    """
    # upper triangle (i<j)
    iu = np.triu_indices(A2.shape[0], k=1)
    x = np.rint(A2[iu]).astype(np.int64)
    x = x[x >= 2]
    if x.size == 0:
        return 0, 0
    total_pairs = int(np.sum(x * (x - 1) // 2))
    c4 = total_pairs // 2
    return int(c4), int(total_pairs)

def _four_cliques_from_A(A: np.ndarray) -> int:
    """
    For n~20, brute force over all 4-sets is cheap (4845 max).
    A is symmetric with 0 diagonal, 0/1 entries.
    A[K,K].sum() counts each undirected edge twice, so K4 => sum == 12.
    """
    n = A.shape[0]
    if n < 4:
        return 0
    cnt = 0
    for a, b, c, d in combinations(range(n), 4):
        sub = A[[a, b, c, d]][:, [a, b, c, d]]
        if sub.sum() == 12.0:
            cnt += 1
    return int(cnt)

def _edge_embeddedness_list_from_A(A: np.ndarray, A2: np.ndarray) -> List[float]:
    """
    For each edge (i,j), number of common neighbors equals A2[i,j]
    for simple 0/1 adjacency.
    """
    iu = np.triu_indices(A.shape[0], k=1)
    edges_mask = A[iu] > 0.0
    if not np.any(edges_mask):
        return []
    # Embeddedness for those edges
    emb = A2[iu][edges_mask]
    return np.asarray(emb, dtype=float).tolist()

# ========================= 5-motifs (raw) =========================
def _five_clique_count(G):
    cnt = 0
    for clq in nx.find_cliques(G):
        k = len(clq)
        if k >= 5:
            cnt += math.comb(k, 5)
    return int(cnt)

def _five_cycle_count(G):
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

# ========================= induced 5-node (WL, g_*_5 naming) =========================
_G5_REPS = []; _G5_HASH2NAME = {}; _G5_NAMES = []

def _init_g5_representatives():
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

        names = []
        for i, rep in enumerate(reps):
            m = rep.number_of_edges()
            degs = sorted([d for _, d in rep.degree()])
            if m == 4:
                if degs == [1,1,1,1,4]: names.append("g_0_5")
                else: names.append(f"g_{i}_5")
            elif m == 5:
                if degs == [1,1,1,2,3]: names.append("g_1_5")
                elif degs == [1,1,2,2,2]: names.append("g_2_5")
                else: names.append(f"g_{i}_5")
            elif m == 6:
                if degs == [1,1,2,2,2]: names.append("g_3_5")
                elif degs == [1,2,2,2,3]: names.append("g_4_5")
                elif degs == [2,2,2,2,2]: names.append("g_7_5")
                else: names.append(f"g_{i}_5")
            elif m == 7:
                if degs == [1,2,2,2,4]: names.append("g_5_5")
                elif degs == [2,2,2,3,3]: names.append("g_6_5")
                elif degs == [2,2,2,2,3]: names.append("g_8_5")
                else: names.append(f"g_{i}_5")
            elif m == 8:
                # For m==8 we don't distinguish all shapes here; keep unique ids
                names.append(f"g_{i}_5")
            elif m == 9:
                names.append("g_17_5")
            elif m == 10:
                names.append("g_20_5")
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

# 10 edges among 5 nodes, fixed order for bitmask
_EDGES5 = [
    (0,1),(0,2),(0,3),(0,4),
    (1,2),(1,3),(1,4),
    (2,3),(2,4),
    (3,4),
]

_IND5_LUT = None  # length 1024; entry is name like "g_7_5" or None if disconnected/unmapped

def _init_ind5_lut():
    """
    Build a lookup table for all 1024 possible 5-node simple graphs:
    mask -> g_*_5 name (only for connected graphs), else None.
    Uses WL hash + fallback isomorphism ONCE at init time.
    """
    global _IND5_LUT
    if _IND5_LUT is not None:
        return
    # If WL atlas reps not available, we can't map reliably
    if (not _HAS_WL) or (not _G5_REPS) or (not _G5_NAMES):
        _IND5_LUT = [None] * 1024
        return

    lut = [None] * 1024

    for mask in range(1024):
        # Build the 5-node graph for this mask
        H = nx.Graph()
        H.add_nodes_from(range(5))
        for bit, (i, j) in enumerate(_EDGES5):
            if (mask >> bit) & 1:
                H.add_edge(i, j)

        if not nx.is_connected(H):
            lut[mask] = None
            continue

        try:
            h = wl_hash(H)
            name = _G5_HASH2NAME.get(h)
            if name is None:
                # Fallback isomorphism search against reps (ONLY 1024 cases, done once)
                for rep, rep_name in zip(_G5_REPS, _G5_NAMES):
                    if nx.is_isomorphic(H, rep):
                        name = rep_name
                        _G5_HASH2NAME[h] = name
                        break
            lut[mask] = name
        except Exception:
            lut[mask] = None

    _IND5_LUT = lut

# Initialize at import time (safe; 1024 graphs only)
try:
    _init_ind5_lut()
except Exception:
    _IND5_LUT = [None] * 1024


def _chunked(iterable, n_chunks):
    it = list(iterable)
    if n_chunks <= 1 or len(it) == 0:
        yield it; return
    size = (len(it) + n_chunks - 1) // n_chunks
    for i in range(0, len(it), size):
        yield it[i:i+size]

def _induced4_counts(G, *, max_samples=None, rng=None, n_jobs: int = 0):
    counts0 = {"K1_3": 0, "P4": 0, "C4": 0, "TailedTriangle": 0, "Diamond": 0, "K4": 0}
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
            key = _classify_induced4(H)
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


def _induced5_counts_bitmask(G: nx.Graph, *, sample_fraction: float = 1.0, random_state: int = 42):
    """
    Returns (counts_dict, connected_total_float).
    Counts only connected induced 5-subgraphs, matching your previous behavior.
    """
    _init_ind5_lut()
    counts = {name: 0 for name in _G5_NAMES}

    n = G.number_of_nodes()
    if n < 5 or _IND5_LUT is None:
        return counts, float("nan")

    nodes = list(G.nodes())
    idx = {u:i for i,u in enumerate(nodes)}

    # neighbor bitmask per node index
    nbr = [0] * n
    for u, v in G.edges():
        iu = idx.get(u); iv = idx.get(v)
        if iu is None or iv is None or iu == iv:
            continue
        nbr[iu] |= (1 << iv)
        nbr[iv] |= (1 << iu)

    rnd = random.Random(random_state)
    connected_total = 0

    # iterate 5-sets (exact) but optionally Bernoulli-sample them like your current code
    for a, b, c, d, e in combinations(range(n), 5):
        if sample_fraction < 1.0 and rnd.random() >= sample_fraction:
            continue

        # Build 10-bit mask in _EDGES5 order for nodes [a,b,c,d,e]
        S = [a, b, c, d, e]
        mask = 0
        for bit, (i, j) in enumerate(_EDGES5):
            u = S[i]; v = S[j]
            if (nbr[u] >> v) & 1:
                mask |= (1 << bit)

        name = _IND5_LUT[mask]
        if name is not None:
            counts[name] += 1
            connected_total += 1

    return counts, float(connected_total)


# ========================= Distances / communities =========================
def _distance_stats_sampled(G, max_sources=200, exact_if_leq=500):
    n = G.number_of_nodes()
    if n == 0:
        return np.nan, np.nan, np.nan
    try:
        if nx.is_connected(G) and n <= exact_if_leq:
            sp = dict(nx.all_pairs_shortest_path_length(G))
            dists = [d for u in sp for d in sp[u].values()]
            ecc   = list(nx.eccentricity(G).values())
        else:
            nodes = list(G.nodes())
            k = min(max_sources, len(nodes))
            seeds = random.sample(nodes, k) if k < len(nodes) else nodes
            dists, ecc = [], []
            for s in seeds:
                lengths = nx.single_source_shortest_path_length(G, s)
                if lengths:
                    dists.extend(lengths.values())
                    ecc.append(max(lengths.values()))
        d = np.asarray(dists, dtype=float); e = np.asarray(ecc, dtype=float)
        d = d[np.isfinite(d)]; e = e[np.isfinite(e)]
        eff_p90 = float(np.quantile(d, 0.90)) if d.size else np.nan
        ecc_mean = float(np.mean(e)) if e.size else np.nan
        ecc_p90  = float(np.quantile(e, 0.90)) if e.size else np.nan
        return eff_p90, ecc_mean, ecc_p90
    except Exception:
        return np.nan, np.nan, np.nan

def _community_features(G):
    base = {"Comm_modularity": np.nan, "Comm_count": np.nan,
            "Comm_size_max": np.nan, "Comm_size_gini": np.nan,
            "Comm_internal_edge_frac": np.nan}
    try:
        import community as community_louvain
        if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
            return base
        part = community_louvain.best_partition(G)
        if not part:
            return base
        sizes = list(Counter(part.values()).values())
        internal = sum(1 for u, v in G.edges() if part.get(u) == part.get(v))
        base["Comm_modularity"] = float(community_louvain.modularity(part, G))
        base["Comm_count"] = float(len(sizes))
        base["Comm_size_max"] = float(max(sizes)) if sizes else np.nan
        base["Comm_size_gini"] = gini(sizes)
        base["Comm_internal_edge_frac"] = float(internal / G.number_of_edges())
        return base
    except Exception:
        return base

# ========================= Single-graph extractor =========================
def _lap_time_grid(tmin: float, tmax: float, points: int) -> List[float]:
    return list(np.logspace(np.log10(tmin), np.log10(tmax), points))

def extract_features_single_graph(
    G: nx.Graph,
    groups: Dict[str, bool],
    *,
    lap_eigs_k: int,
    netlsd_points: int,
    netlsd_tmin: float,
    netlsd_tmax: float,
    ind4_exact_nmax: int,
    ind4_max_samples: int,
    ind5_sample_frac: float,
    distance_max_sources: int,
    distance_exact_if_leq: int,
    seed: int,
    intra_workers: int,
    betweenness_k: int,
) -> Dict[str, Any]:

    rnd = random.Random(seed)
    enabled = lambda k: groups.get(k, True) if groups else True
    feats: Dict[str, Any] = {}
    n = G.number_of_nodes(); m = G.number_of_edges()

    # BASIC
    degs = [d for _, d in G.degree()]
    if enabled("basic"):
        dh = Counter(degs)
        feats.update({
            "Basic_num_nodes": n, "Basic_num_edges": m,
            "Basic_min_degree": min(degs) if degs else np.nan,
            "Basic_max_degree": max(degs) if degs else np.nan,
            "Basic_avg_degree": float(np.mean(degs)) if degs else np.nan,
            "Basic_degree_std": float(np.std(degs))  if degs else np.nan,
            "Basic_degree_skew": float(pd.Series(degs).skew()) if len(degs) > 2 else np.nan,
            "Basic_density": nx.density(G),
            "Basic_edge_to_node_ratio": m / n if n else np.nan,
            "Basic_degree_entropy": shannon_entropy(dh),
        })
        feats["Assortativity_degree"] = _safe_degree_assortativity(G)
        try:
            cc = nx.clustering(G); cc_vals = list(cc.values())
        except Exception:
            cc_vals = []
        feats["Clustering_mean"] = float(np.mean(cc_vals)) if cc_vals else np.nan
        feats["Clustering_q10"]  = _q(cc_vals, 0.10)
        feats["Clustering_q50"]  = _q(cc_vals, 0.50)
        feats["Clustering_q90"]  = _q(cc_vals, 0.90)
        feats["Clustering_frac_zero"] = _frac(cc_vals, lambda a: np.isclose(a, 0.0))
        feats["Clustering_frac_one"]  = _frac(cc_vals, lambda a: np.isclose(a, 1.0))
        feats["Degree_gini"] = gini(degs)

    # CONNECTIVITY + distances
    if enabled("connectivity"):
        feats.update({
            "Connectivity_is_connected": try_or_nan(nx.is_connected, G),
            "Connectivity_num_components": try_or_nan(nx.number_connected_components, G),
            "Connectivity_diameter": try_or_nan(nx.diameter, G),
            "Connectivity_radius": try_or_nan(nx.radius, G),
            "Connectivity_avg_shortest_path_length": try_or_nan(nx.average_shortest_path_length, G),
            "Connectivity_wiener_index": try_or_nan(nx.wiener_index, G),
        })
        eff_p90, ecc_mean, ecc_p90 = _distance_stats_sampled(
            G, max_sources=distance_max_sources, exact_if_leq=distance_exact_if_leq
        )
        feats["Eff_diameter_p90"] = eff_p90
        feats["Ecc_mean"] = ecc_mean
        feats["Ecc_q90"]  = ecc_p90

    # CENTRALITY
    if enabled("centrality"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if betweenness_k and betweenness_k > 0:
                try:
                    bc = nx.betweenness_centrality(G, k=min(betweenness_k, max(1, G.number_of_nodes())), normalized=True, seed=seed)
                except TypeError:
                    bc = try_or_nan(nx.betweenness_centrality, G, normalized=True)
            else:
                bc = try_or_nan(nx.betweenness_centrality, G, normalized=True)
            cc = try_or_nan(nx.closeness_centrality,  G)
            try:
                ec = nx.eigenvector_centrality_numpy(G) if m > 0 else {}
            except Exception:
                ec = {}
        def _stats(d):
            if isinstance(d, dict) and d:
                vals = list(d.values())
                return {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "std": float(np.std(vals)),
                        "skew": float(pd.Series(vals).skew()) if len(vals) > 2 else np.nan}
            return {"mean": np.nan, "max": np.nan, "std": np.nan, "skew": np.nan}
        feats.update({f"Centrality_betweenness_{k}": v for k, v in _stats(bc).items()})
        feats.update({f"Centrality_closeness_{k}":  v for k, v in _stats(cc).items()})
        feats.update({f"Centrality_eigenvector_{k}": v for k, v in _stats(ec).items()})

    # CORE
    if enabled("core"):
        core_numbers = try_or_nan(nx.core_number, G)
        if isinstance(core_numbers, dict) and core_numbers:
            core_vals = list(core_numbers.values())
            feats["Core_max_core_index"]  = max(core_vals)
            feats["Core_core_index_mean"] = float(np.mean(core_vals))
        else:
            feats["Core_max_core_index"]  = np.nan
            feats["Core_core_index_mean"] = np.nan

    # ROBUSTNESS
    if enabled("robustness"):
        feats["Robust_articulation_points"] = try_or_nan(lambda g: len(list(nx.articulation_points(g))), G)
        feats["Robust_bridge_count"]        = try_or_nan(lambda g: len(list(nx.bridges(g))),            G)

    # CYCLES
    if enabled("cycles"):
        cbasis  = nx.cycle_basis(G)
        feats["Cycle_num_cycles_len_5"] = sum(1 for c in cbasis if len(c) == 5)
        feats["Cycle_num_cycles_len_6"] = sum(1 for c in cbasis if len(c) == 6)

    # SPECTRAL (Laplacian)
    leigs = None
    if enabled("spectral_laplacian"):
        try:
            leigs = _laplacian_eigs(G)
            feats["Spectral_algebraic_connectivity"] = float(leigs[1])           if len(leigs) > 1 else np.nan
            feats["Spectral_spectral_gap"]           = float(leigs[1]-leigs[0]) if len(leigs) > 1 else np.nan
            feats["Spectral_laplacian_mean"]         = float(np.mean(leigs)) if leigs.size else np.nan
            feats["Spectral_laplacian_std"]          = float(np.std(leigs))  if leigs.size else np.nan
            feats["Spectral_laplacian_skew"]         = float(pd.Series(leigs).skew()) if leigs.size > 2 else np.nan
            K = max(1, int(lap_eigs_k))
            pad = np.full(K, np.nan); pad[:min(K, len(leigs))] = leigs[:K]
            feats.update({f"Spectral_lap_eig_{i}": float(pad[i]) for i in range(K)})
            nonzero = leigs[leigs > 1e-12]
            feats["Kirchhoff_index"] = float(n * np.sum(1.0 / nonzero)) if len(nonzero) >= 1 else np.nan
            feats["Spectral_kirchhoff_index"] = feats["Kirchhoff_index"]
        except Exception:
            K = max(1, int(lap_eigs_k))
            feats.update({
                "Spectral_algebraic_connectivity": np.nan,
                "Spectral_spectral_gap":           np.nan,
                "Spectral_laplacian_mean":         np.nan,
                "Spectral_laplacian_std":          np.nan,
                "Spectral_laplacian_skew":         np.nan,
                **{f"Spectral_lap_eig_{i}": np.nan for i in range(K)},
                "Kirchhoff_index": np.nan,
                "Spectral_kirchhoff_index": np.nan,
            })
        # Heat traces at multiple times
        feats.update(_laplacian_heat_traces(leigs, ts=(0.1, 0.5, 1.0, 2.0, 5.0)))

    # NETLSD summaries
    if enabled("netlsd"):
        if leigs is None:
            try:
                leigs = _laplacian_eigs(G)
            except Exception:
                leigs = np.array([])
        if leigs is not None and len(leigs) > 0 and not np.any(np.isnan(leigs)):
            ts = _lap_time_grid(netlsd_tmin, netlsd_tmax, netlsd_points)
            heat = np.exp(-np.outer(ts, leigs)).sum(axis=1)
            feats["NetLSD_mean"] = float(np.mean(heat))
            feats["NetLSD_std"]  = float(np.std(heat))
            feats["NetLSD_q10"]  = float(np.quantile(heat, 0.10))
            feats["NetLSD_q90"]  = float(np.quantile(heat, 0.90))
        else:
            feats["NetLSD_mean"] = np.nan
            feats["NetLSD_std"]  = np.nan
            feats["NetLSD_q10"]  = np.nan
            feats["NetLSD_q90"]  = np.nan

    # PLANARITY
    if enabled("planarity"):
        try:
            planar, emb = nx.check_planarity(G)
            # Planarity_is_planar removed per user request
            # feats["Planarity_is_planar"] = bool(planar)
            if planar:
                face_list = extract_faces(emb)
                f_sizes = [len(face) for face in face_list]
                feats["Planarity_num_faces"]      = float(len(face_list))
                feats["Planarity_face_size_mean"] = float(np.mean(f_sizes)) if f_sizes else np.nan
                feats["Planarity_face_size_max"]  = float(np.max(f_sizes))  if f_sizes else np.nan
            else:
                feats["Planarity_num_faces"]      = np.nan
                feats["Planarity_face_size_mean"] = np.nan
                feats["Planarity_face_size_max"]  = np.nan
        except Exception:
            # Planarity_is_planar removed per user request
            # feats["Planarity_is_planar"] = np.nan
            feats["Planarity_num_faces"]      = np.nan
            feats["Planarity_face_size_mean"] = np.nan
            feats["Planarity_face_size_max"]  = np.nan

    # SYMMETRY
    if enabled("symmetry"):
        try:
            GM       = nx.algorithms.isomorphism.GraphMatcher(G, G)
            aut_size = len(list(GM.isomorphisms_iter()))
            orbits   = compute_node_orbits(G)
            feats["Symmetry_automorphism_group_order"] = float(aut_size)
            feats["Symmetry_num_orbits"]               = float(len(orbits))
            feats["Symmetry_orbit_size_max"]           = float(max(len(o) for o in orbits)) if orbits else np.nan
        except Exception:
            feats["Symmetry_automorphism_group_order"] = np.nan
            feats["Symmetry_num_orbits"]               = np.nan
            feats["Symmetry_orbit_size_max"]           = np.nan

    # COMMUNITY
    if enabled("community"):
        feats.update(_community_features(G))

    # MOTIFS (3/4) + edge embeddedness stats
    A = None
    if enabled("motifs34") or enabled("spectral_adjacency") or enabled("motifs5") or enabled("induced4") or enabled("induced5"):
        A = _adjacency_matrix(G)
    if enabled("motifs34"):
        # OLDER COMPUTE OVERHEAD STUFF
        #tri = _triangle_count(G, A=A)
        #feats["Motif_triangles"]   = tri
        #feats["Motif_wedges"]      = _wedge_count(G, degs=[d for _, d in G.degree()], tri=tri)
        #feats["Motif_4_cycles"]    = _four_cycle_count(G, A=A)
        #feats["Motif_4_cliques"]   = _four_clique_count(G)
        #m_mean, m_std = _edge_triangle_incidence_stats(G)
        #feats["Motif_triangle_edge_incidence_mean"] = m_mean
        #feats["Motif_triangle_edge_incidence_std"]  = m_std
        #feats["Motif_square_clustering_proxy"]      = _square_clustering_proxy(G, A=A)
        #emb_list = [len(set(G[u]) & set(G[v])) for u, v in G.edges()]
        #feats["Motif_triangle_edge_incidence_median"] = _q(emb_list, 0.50)
        #feats["Motif_triangle_edge_incidence_q90"]    = _q(emb_list, 0.90)
        #feats["Motif_triangle_edge_frac_zero"]        = _frac(emb_list, lambda a: np.isclose(a, 0.0))
        #feats["Motif_triangle_edge_frac_ge2"]         = _frac(emb_list, lambda a: a >= 2)
    # Ensure A is float ndarray 0/1
        if A is None:
            A = _adjacency_matrix(G)

        # Precompute powers once
        A2 = A @ A

        tri = _triangles_from_A(A)
        feats["Motif_triangles"] = tri

        # wedges from degrees: sum C(d,2) - 3*tri
        degs_arr = np.rint(A.sum(axis=1)).astype(np.int64)
        wedges = int(np.sum(degs_arr * (degs_arr - 1) // 2) - 3 * tri)
        feats["Motif_wedges"] = wedges

        c4, denom = _four_cycles_and_denom_from_A2(A2)
        feats["Motif_4_cycles"] = c4

        feats["Motif_4_cliques"] = _four_cliques_from_A(A)

        # Edge embeddedness (= common neighbors) list for all edges
        emb_list = _edge_embeddedness_list_from_A(A, A2)

        feats["Motif_triangle_edge_incidence_mean"]   = float(np.mean(emb_list)) if emb_list else 0.0
        feats["Motif_triangle_edge_incidence_std"]    = float(np.std(emb_list, ddof=0)) if emb_list else 0.0
        feats["Motif_triangle_edge_incidence_median"] = _q(emb_list, 0.50)
        feats["Motif_triangle_edge_incidence_q90"]    = _q(emb_list, 0.90)
        feats["Motif_triangle_edge_frac_zero"]        = _frac(emb_list, lambda a: np.isclose(a, 0.0))
        feats["Motif_triangle_edge_frac_ge2"]         = _frac(emb_list, lambda a: a >= 2)

        # Keep your existing proxy definition, but compute from A2/denom exactly
        feats["Motif_square_clustering_proxy"] = (float(4.0 * c4 / denom) if denom else np.nan)

    # MOTIFS (5)
    if enabled("motifs5"):
        feats["Motif_5_cycles"]  = _five_cycle_count(G)
        feats["Motif_5_cliques"] = _five_clique_count(G)

    # INDUCED 4
    if enabled("induced4"):
        max_samples = None
        if n > ind4_exact_nmax:
            max_samples = ind4_max_samples if ind4_max_samples > 0 else 50000
        #ind4 = _induced4_counts(G, max_samples=max_samples, rng=rnd, n_jobs=intra_workers)
        ind4 = _induced4_counts_bitmask(G, max_samples=max_samples, rng=rnd)

        # Original naming
        for k, v in ind4.items():
            feats[f"Motif_induced_{k}"] = float(v)
        # g_*_4 naming mapped from original
        g_map = {
            "P4": "g_1_4",
            "K1_3": "g_2_4",
            "C4": "g_3_4",
            "TailedTriangle": "g_4_4",
            "Diamond": "g_5_4",
            "K4": "g_6_4",
        }
        for old_name, g_name in g_map.items():
            feats[f"Motif_induced_{g_name}"] = float(ind4.get(old_name, 0.0))
        total_4sets = safe_comb(n, 4) if n >= 4 else np.nan
        feats["Motif_induced_connected_per_4set"] = float(sum(ind4.values()) / total_4sets) if (total_4sets and total_4sets==total_4sets) else np.nan

    # INDUCED 5 (g_*_5)
    if enabled("induced5"):
        #old slow
        #ind5_counts, connected5 = _induced5_counts(G, sample_fraction=ind5_sample_frac, random_state=seed, n_jobs=intra_workers)
        ind5_counts, connected5 = _induced5_counts_bitmask(G, sample_fraction=ind5_sample_frac, random_state=seed)

        for name, val in ind5_counts.items():
            feats[f"Motif_induced5_{name}"] = float(val)
        total_5sets = safe_comb(n, 5) if n >= 5 else np.nan
        feats["Motif_induced_connected_per_5set"] = (connected5 / total_5sets) if (total_5sets and total_5sets==total_5sets and connected5==connected5) else np.nan

    # SPECTRAL (Adjacency/Energy)
    if enabled("spectral_adjacency"):
        if A is None:
            A = _adjacency_matrix(G)
        energy, estrada, m2, m3, m4 = _adjacency_spectrum_features(A)
        feats["Adjacency_energy"]        = energy
        feats["Adjacency_estrada_index"] = estrada
        feats["Adjacency_moment_2"]      = m2
        feats["Adjacency_moment_3"]      = m3
        feats["Adjacency_moment_4"]      = m4
        # Duplicate names with Spectral_ prefix
        feats["Spectral_adjacency_energy"]        = energy
        feats["Spectral_adjacency_estrada_index"] = estrada
        feats["Spectral_adjacency_moment_2"]      = m2
        feats["Spectral_adjacency_moment_3"]      = m3
        feats["Spectral_adjacency_moment_4"]      = m4

    # TDA
    if enabled("tda"):
        feats.update(_tda_vr_h_summary_from_spdm(G))

    # NORMALIZATIONS
    nf = float(n); mf = float(m)
    avgdeg = feats.get("Basic_avg_degree", np.nan)

    if enabled("basic") or enabled("connectivity"):
        feats["Basic_avg_degree_norm"]       = feats.get("Basic_avg_degree", np.nan) / max(nf-1, 1) if nf==nf else np.nan
        feats["Basic_degree_entropy_norm"]   = feats.get("Basic_degree_entropy", np.nan) / np.log2(max(nf-1, 2)) if nf==nf else np.nan
        for col, denom in [("Connectivity_diameter", nf-1), ("Connectivity_radius", nf-1)]:
            v = feats.get(col, np.nan)
            feats[f"{col}_norm"] = v / max(denom, 1) if (v==v and nf==nf) else np.nan
        pairs = nf*(nf-1)/2.0 if nf==nf else np.nan
        wd = feats.get("Connectivity_wiener_index", np.nan)
        feats["Wiener_mean_distance"] = wd/pairs if (wd==wd and pairs and pairs==pairs) else np.nan
        feats["Connectivity_num_components_per_node"] = feats.get("Connectivity_num_components", np.nan)/nf if nf==nf else np.nan

    if enabled("robustness"):
        feats["Robust_articulation_points_per_node"] = feats.get("Robust_articulation_points", np.nan)/nf if nf==nf else np.nan
        feats["Robust_bridge_count_per_edge"]        = feats.get("Robust_bridge_count", np.nan)/mf if (mf==mf and mf) else np.nan

    if enabled("motifs34") or enabled("motifs5") or enabled("induced4") or enabled("induced5"):
        Cn3 = safe_comb(int(nf),3); Cn4 = safe_comb(int(nf),4); Cn5 = safe_comb(int(nf),5)
        if enabled("motifs34"):
            tri = feats.get("Motif_triangles", np.nan)
            c4  = feats.get("Motif_4_cycles", np.nan)
            k4  = feats.get("Motif_4_cliques", np.nan)
            w   = feats.get("Motif_wedges", np.nan)
            feats["Motif_triangles_per_Cn3"] = tri/Cn3 if Cn3 and Cn3==Cn3 else np.nan
            feats["Motif_4_cycles_per_Cn4"]  = c4/Cn4  if Cn4 and Cn4==Cn4 else np.nan
            feats["Motif_4_cliques_per_Cn4"] = k4/Cn4  if Cn4 and Cn4==Cn4 else np.nan
            max_wedges = nf*((nf-1)*(nf-2)/2.0) if (nf==nf and nf>=3) else np.nan
            feats["Motif_wedges_per_max"]    = w/max_wedges if (max_wedges and max_wedges==max_wedges) else np.nan
            for key in ["K1_3","P4","C4","TailedTriangle","Diamond","K4"]:
                raw = feats.get(f"Motif_induced_{key}", np.nan)
                feats[f"Motif_induced_{key}_per_Cn4"] = raw/Cn4 if Cn4 and Cn4==Cn4 else np.nan
            # g_*_4 per_Cn4
            for old, gname in {"P4":"g_1_4","K1_3":"g_2_4","C4":"g_3_4","TailedTriangle":"g_4_4","Diamond":"g_5_4","K4":"g_6_4"}.items():
                raw = feats.get(f"Motif_induced_{gname}", np.nan)
                feats[f"Motif_induced_{gname}_per_Cn4"] = raw/Cn4 if Cn4 and Cn4==Cn4 else np.nan
        if enabled("motifs5"):
            c5 = feats.get("Motif_5_cycles", np.nan)
            k5 = feats.get("Motif_5_cliques", np.nan)
            feats["Motif_5_cycles_per_Cn5"]  = c5/Cn5 if Cn5 and Cn5==Cn5 else np.nan
            feats["Motif_5_cliques_per_Cn5"] = k5/Cn5 if Cn5 and Cn5==Cn5 else np.nan
            max_c5 = (Cn5 * 12) if Cn5 and Cn5==Cn5 else np.nan
            feats["Motif_5_cycles_per_Kn"]   = c5/max_c5 if (max_c5 and max_c5==max_c5) else np.nan
        if enabled("induced5") and Cn5 and Cn5==Cn5:
            for name in _G5_NAMES:
                raw5 = feats.get(f"Motif_induced5_{name}", np.nan)
                feats[f"Motif_induced5_{name}_per_Cn5"] = raw5 / Cn5 if raw5 == raw5 else np.nan

    if enabled("centrality") and avgdeg==avgdeg and nf==nf:
        feats["Centrality_closeness_mean_norm"] = feats.get("Centrality_closeness_mean", np.nan)*(nf-1)
        feats["Centrality_closeness_max_norm"]  = feats.get("Centrality_closeness_max",  np.nan)*(nf-1)

    if enabled("spectral_laplacian"):
        sac = feats.get("Spectral_algebraic_connectivity", np.nan)
        lm  = feats.get("Spectral_laplacian_mean", np.nan)
        feats["Spectral_algebraic_connectivity_over_avgdeg"] = sac/max(avgdeg,1e-9) if (sac==sac and avgdeg==avgdeg) else np.nan
        feats["Spectral_spectral_gap_rel"] = feats.get("Spectral_spectral_gap", np.nan)/max(lm,1e-9) if lm==lm else np.nan
        for t in ("0.1","1.0","5.0"):
            col = f"Spectral_laplacian_heat_trace_t{t}"
            if col in feats and nf==nf and feats[col]==feats[col]:
                feats[f"{col}_per_node"] = feats[col]/nf

    if enabled("spectral_adjacency"):
        ae = feats.get("Adjacency_estrada_index", np.nan)
        en = feats.get("Adjacency_energy", np.nan)
        feats["Adjacency_energy_per_node"]   = en/nf if (en==en and nf==nf) else np.nan
        feats["Adjacency_energy_over_fro"]   = en/math.sqrt(2.0*max(mf,1e-12)) if (en==en and mf==mf) else np.nan
        feats["Adjacency_estrada_per_node"]  = ae/nf if (ae==ae and nf==nf) else np.nan
        feats["log_Adjacency_estrada_per_node"] = np.log(max(ae if ae==ae else 1.0,1.0))/nf if nf==nf else np.nan
        if avgdeg==avgdeg:
            feats["Adjacency_moment_2_over_avgdeg"]  = feats.get("Adjacency_moment_2", np.nan) / max(avgdeg,1e-9)
            feats["Adjacency_moment_3_over_avgdeg3"] = feats.get("Adjacency_moment_3", np.nan) / max(avgdeg**3,1e-9)
            feats["Adjacency_moment_4_over_avgdeg4"] = feats.get("Adjacency_moment_4", np.nan) / max(avgdeg**4,1e-9)

    if enabled("planarity") and nf==nf:
        upper = 2*float(nf) - 4.0 if nf >= 3 else np.nan
        faces = feats.get("Planarity_num_faces", np.nan)
        feats["Planarity_num_faces_over_upperbound"] = faces/upper if (faces==faces and upper==upper and upper) else np.nan
        feats["Planarity_face_size_mean_norm"] = feats.get("Planarity_face_size_mean", np.nan)/float(nf) if nf==nf else np.nan

    if enabled("symmetry") and nf==nf:
        try:
            aut = feats.get("Symmetry_automorphism_group_order", np.nan)
            log_aut  = np.log(max(aut,1.0)) if aut==aut else np.nan
            log_nfact = math.lgamma(float(nf)+1.0)
            feats["Symmetry_aut_size_log_over_log_nfact"] = (log_aut / max(log_nfact,1e-9)) if (log_aut==log_aut and log_nfact==log_nfact) else np.nan
            feats["Symmetry_num_orbits_per_node"]    = feats.get("Symmetry_num_orbits", np.nan)/nf
            feats["Symmetry_orbit_size_max_per_node"]= feats.get("Symmetry_orbit_size_max", np.nan)/nf
        except Exception:
            feats["Symmetry_aut_size_log_over_log_nfact"] = np.nan
            feats["Symmetry_num_orbits_per_node"] = np.nan
            feats["Symmetry_orbit_size_max_per_node"] = np.nan

    if enabled("tda") and nf==nf:
        for H in ["H0","H1"]:
            cnt = feats.get(f"TDA_{H}_count", np.nan)
            if cnt==cnt:
                feats[f"TDA_{H}_count_per_node"] = cnt/nf
            for stat in ["total_persistence","mean_persistence","max_persistence","mean_birth","mean_death"]:
                col = f"TDA_{H}_{stat}"
                val = feats.get(col, np.nan)
                feats[col + "_over_diam"] = val / max(float(nf)-1.0,1.0) if val==val else np.nan
        for q in ["q25","q50","q75"]:
            c0 = f"TDA_Betti0_at_{q}"; c1 = f"TDA_Betti1_at_{q}"
            if c0 in feats and feats[c0]==feats[c0]:
                feats[c0 + "_per_node"] = feats[c0]/nf
            if c1 in feats and feats[c1]==feats[c1]:
                feats[c1 + "_per_node"] = feats[c1]/nf

    return feats

# ========================= Multi-layer wrappers =========================
try:
    import orjson
    _HAS_ORJSON = True
except Exception:
    _HAS_ORJSON = False

def parse_edges_fast(s: str):
    if not s:
        return []
    s = s.strip()
    if s == "" or s == "[]" or s.lower() in ("nan","none","null"):
        return []
    # convert Python tuple syntax to JSON list syntax
    j = s.replace("(", "[").replace(")", "]").replace("'", '"')
    return orjson.loads(j)

def parse_edge_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple)):
        return [tuple(e) for e in x]
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return []
        # FAST PATH
        if _HAS_ORJSON:
            try:
                obj = parse_edges_fast(s)
                if isinstance(obj, list):
                    return [tuple(e) for e in obj]
            except Exception:
                pass
        # FALLBACK
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [tuple(e) for e in obj]
        except Exception:
            return []
    return []

def build_graphs_from_row(row):
    if "EDGES" in row and pd.notna(row["EDGES"]):
        E = parse_edge_list(row["EDGES"])
        G = nx.Graph(); G.add_edges_from(E)
        return {"SINGLE": G}

    den_list = parse_edge_list(row.get("DEN_EDGES")) if "DEN_EDGES" in row else []
    num_list = parse_edge_list(row.get("NUM_EDGES")) if "NUM_EDGES" in row else []

    has_den = len(den_list) > 0
    has_num = len(num_list) > 0

    if has_den and has_num:
        G_DEN, G_NUM, G_TOTAL = nx.Graph(), nx.Graph(), nx.Graph()
        G_DEN.add_edges_from(den_list); G_NUM.add_edges_from(num_list)
        G_TOTAL.add_edges_from(den_list + num_list)
        return {"DEN": G_DEN, "NUM": G_NUM, "TOTAL": G_TOTAL}

    if has_den:
        G_DEN = nx.Graph(); G_DEN.add_edges_from(den_list)
        return {"DEN": G_DEN}

    if has_num:
        G_NUM = nx.Graph(); G_NUM.add_edges_from(num_list)
        return {"NUM": G_NUM}

    return {"SINGLE": nx.Graph()}

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
        total_c4 = _four_cycle_count(G_total)
        c4_den   = _four_cycle_count(G_den)
        c4_num   = _four_cycle_count(G_num)
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

def extract_features_row(
    row: pd.Series,
    groups: Dict[str, bool],
    *,
    lap_eigs_k: int,
    netlsd_points: int,
    netlsd_tmin: float,
    netlsd_tmax: float,
    ind4_exact_nmax: int,
    ind4_max_samples: int,
    ind5_sample_frac: float,
    distance_max_sources: int,
    distance_exact_if_leq: int,
    seed: int,
    intra_workers: int,
    betweenness_k: int,
) -> Dict[str, Any]:
    graphs = build_graphs_from_row(row)

    # Single-graph path (EDGES or empty)
    if "SINGLE" in graphs:
        feats = extract_features_single_graph(
            graphs["SINGLE"], groups,
            lap_eigs_k=lap_eigs_k, netlsd_points=netlsd_points,
            netlsd_tmin=netlsd_tmin, netlsd_tmax=netlsd_tmax,
            ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
            ind5_sample_frac=ind5_sample_frac,
            distance_max_sources=distance_max_sources,
            distance_exact_if_leq=distance_exact_if_leq,
            seed=seed, intra_workers=intra_workers, betweenness_k=betweenness_k,
        )
        for meta_col in ("COEFFICIENTS", "Unnamed: 0"):
            if meta_col in row:
                feats[meta_col] = row[meta_col]
        return feats

    # Layered path (DEN/NUM[/TOTAL]), possibly partial
    out: Dict[str, Any] = {}
    
    has_den = "DEN" in graphs
    has_num = "NUM" in graphs
    has_both = has_den and has_num
    
    # Check if we have DEN_EDGES or NUM_EDGES columns (even if empty)
    # This determines if we should use suffixed features
    # We check if the column exists in the row, regardless of whether it has a value
    has_den_edges_col = "DEN_EDGES" in row.index
    has_num_edges_col = "NUM_EDGES" in row.index
    has_layered_columns = has_den_edges_col or has_num_edges_col

    if has_layered_columns:
        # We have DEN_EDGES or NUM_EDGES columns: use suffixed features
        # Only create features for graphs that actually exist (non-empty)
        if has_den:
            feats_den = extract_features_single_graph(
                graphs["DEN"], groups,
                lap_eigs_k=lap_eigs_k, netlsd_points=netlsd_points,
                netlsd_tmin=netlsd_tmin, netlsd_tmax=netlsd_tmax,
                ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
                ind5_sample_frac=ind5_sample_frac,
                distance_max_sources=distance_max_sources,
                distance_exact_if_leq=distance_exact_if_leq,
                seed=seed, intra_workers=intra_workers, betweenness_k=betweenness_k,
            )
            out.update({f"{k}_DEN": v for k, v in feats_den.items()})
        
        if has_num:
            feats_num = extract_features_single_graph(
                graphs["NUM"], groups,
                lap_eigs_k=lap_eigs_k, netlsd_points=netlsd_points,
                netlsd_tmin=netlsd_tmin, netlsd_tmax=netlsd_tmax,
                ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
                ind5_sample_frac=ind5_sample_frac,
                distance_max_sources=distance_max_sources,
                distance_exact_if_leq=distance_exact_if_leq,
                seed=seed, intra_workers=intra_workers, betweenness_k=betweenness_k,
            )
            out.update({f"{k}_NUM": v for k, v in feats_num.items()})
        
        # Only create TOTAL if both DEN and NUM exist
        if has_both and "TOTAL" in graphs:
            feats_total = extract_features_single_graph(
                graphs["TOTAL"], groups,
                lap_eigs_k=lap_eigs_k, netlsd_points=netlsd_points,
                netlsd_tmin=netlsd_tmin, netlsd_tmax=netlsd_tmax,
                ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
                ind5_sample_frac=ind5_sample_frac,
                distance_max_sources=distance_max_sources,
                distance_exact_if_leq=distance_exact_if_leq,
                seed=seed, intra_workers=intra_workers, betweenness_k=betweenness_k,
            )
            out.update({f"{k}_TOTAL": v for k, v in feats_total.items()})
    else:
        # Single-layer graph (EDGES column): compute features without suffix
        # This shouldn't happen here since SINGLE is handled above, but keep for safety
        for tag in ("DEN", "NUM"):
            if tag in graphs:
                feats = extract_features_single_graph(
                    graphs[tag], groups,
                    lap_eigs_k=lap_eigs_k, netlsd_points=netlsd_points,
                    netlsd_tmin=netlsd_tmin, netlsd_tmax=netlsd_tmax,
                    ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
                    ind5_sample_frac=ind5_sample_frac,
                    distance_max_sources=distance_max_sources,
                    distance_exact_if_leq=distance_exact_if_leq,
                    seed=seed, intra_workers=intra_workers, betweenness_k=betweenness_k,
                )
                out.update(feats)
                break

    if groups.get("cross_layer", False) and ("DEN" in graphs) and ("NUM" in graphs):
        G_DEN, G_NUM = graphs["DEN"], graphs["NUM"]
        G_TOTAL = nx.compose(G_DEN, G_NUM)
        n = G_TOTAL.number_of_nodes()
        e_den, e_num = G_DEN.number_of_edges(), G_NUM.number_of_edges()
        union_edges = set(G_DEN.edges()) | set(G_NUM.edges())
        inter_edges = set(G_DEN.edges()) & set(G_NUM.edges())
        # Edge_ratio_DEN_NUM and Edge_overlap_frac removed per user request
        # out["Edge_ratio_DEN_NUM"] = e_den / max(e_num, 1)
        out["Edge_overlap_count"] = len(inter_edges)
        out["Edge_jaccard"] = len(inter_edges) / max(len(union_edges), 1)
        # out["Edge_overlap_frac_DEN"] = len(inter_edges) / max(e_den, 1)
        # out["Edge_overlap_frac_NUM"] = len(inter_edges) / max(e_num, 1)
        out["Node_overlap_frac"] = len(set(G_DEN.nodes()) & set(G_NUM.nodes())) / max(n, 1)

        cross_tri = count_cross_layer_motifs(G_DEN, G_NUM, "triangle")
        cross_c4  = count_cross_layer_motifs(G_DEN, G_NUM, "4cycle")
        cross_dia = count_cross_layer_motifs(G_DEN, G_NUM, "diamond")
        cross_k4  = count_cross_layer_motifs(G_DEN, G_NUM, "4clique")
        out["Cross_triangles_mixed"] = cross_tri
        out["Cross_4cycles_mixed"]   = cross_c4
        out["Cross_diamonds_mixed"]  = cross_dia
        out["Cross_K4_mixed"]        = cross_k4

        tri_total = out.get("Motif_triangles_TOTAL", np.nan)
        c4_total  = out.get("Motif_4_cycles_TOTAL", np.nan)
        dia_total = out.get("Motif_induced_Diamond_TOTAL", np.nan)
        k4_total  = out.get("Motif_induced_K4_TOTAL", np.nan)
        out["Cross_triangles_mixed_frac"] = cross_tri / tri_total if tri_total and tri_total == tri_total else np.nan
        out["Cross_4cycles_mixed_frac"]   = cross_c4  / c4_total  if c4_total  and c4_total  == c4_total  else np.nan
        out["Cross_diamonds_mixed_frac"]  = cross_dia / dia_total if dia_total and dia_total == dia_total else np.nan
        out["Cross_K4_mixed_frac"]        = cross_k4  / k4_total  if k4_total  and k4_total  == k4_total  else np.nan

        Cn3 = math.comb(n, 3) if n >= 3 else np.nan
        Cn4 = math.comb(n, 4) if n >= 4 else np.nan
        out["Cross_triangles_mixed_per_Cn3"] = cross_tri / Cn3 if Cn3 and Cn3 == Cn3 else np.nan
        out["Cross_4cycles_mixed_per_Cn4"]   = cross_c4  / Cn4 if Cn4 and Cn4 == Cn4 else np.nan
        out["Cross_diamonds_mixed_per_Cn4"]  = cross_dia / Cn4 if Cn4 and Cn4 == Cn4 else np.nan
        out["Cross_K4_mixed_per_Cn4"]        = cross_k4  / Cn4 if Cn4 and Cn4 == Cn4 else np.nan

    for meta_col in ("COEFFICIENTS", "Unnamed: 0"):
        if meta_col in row:
            out[meta_col] = row[meta_col]
    return out

# ========================= Batch core =========================
def _safe_extract_row(row, **kwargs):
    try:
        return extract_features_row(row, **kwargs)
    except Exception as e:
        return {"__error__": str(e)}

def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all DESIRED_COLUMNS are present, add missing ones as NaN,
    but keep all computed columns (including suffixed ones like _DEN, _NUM, _TOTAL).
    
    If we have layered graphs (suffixed columns), we don't add unsuffixed DESIRED_COLUMNS.
    If we have single-layer graphs (unsuffixed columns), we add missing DESIRED_COLUMNS.
    
    Order: DESIRED_COLUMNS first (if applicable), then any additional computed columns.
    """
    # Check if we have layered graphs (suffixed columns)
    has_suffixed = any(col.endswith(('_DEN', '_NUM', '_TOTAL')) for col in df.columns)
    
    if has_suffixed:
        # Layered graphs: don't add unsuffixed DESIRED_COLUMNS, just keep what we computed
        # Remove any unsuffixed columns that are in DESIRED_COLUMNS (they shouldn't be there)
        # But preserve meta columns like COEFFICIENTS and Unnamed: 0
        meta_cols = {"COEFFICIENTS", "Unnamed: 0"}
        cols_to_keep = [col for col in df.columns if not (col in DESIRED_COLUMNS and col not in meta_cols and not col.endswith(('_DEN', '_NUM', '_TOTAL')))]
        df = df[cols_to_keep]
        # Sort columns: meta columns first, then suffixed columns grouped by feature name, then any extras
        meta_cols_present = [col for col in meta_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in meta_cols]
        all_cols = meta_cols_present + sorted(other_cols)
        return df.reindex(columns=all_cols)
    else:
        # Single-layer graphs: add missing DESIRED_COLUMNS as NaN
        missing = [col for col in DESIRED_COLUMNS if col not in df.columns]
        
        if missing:
            # Create a DataFrame of all missing columns at once
            missing_block = pd.DataFrame(
                {col: np.nan for col in missing},
                index=df.index,
            )
            df = pd.concat([df, missing_block], axis=1)
        
        # Get all columns: DESIRED_COLUMNS first, then any extras
        desired_set = set(DESIRED_COLUMNS)
        extra_cols = [col for col in df.columns if col not in desired_set]
        extra_cols.sort()
        all_cols = DESIRED_COLUMNS + extra_cols
        
        return df.reindex(columns=all_cols)



def compute_batch_df(
    df: pd.DataFrame,
    *,
    groups: Dict[str, bool],
    lap_eigs_k: int,
    netlsd_points: int,
    netlsd_tmin: float,
    netlsd_tmax: float,
    ind4_exact_nmax: int,
    ind4_max_samples: int,
    ind5_sample_frac: float,
    distance_max_sources: int,
    distance_exact_if_leq: int,
    seed: int,
    intra_workers: int,
    betweenness_k: int,
    workers: int,
    backend: str,
    progress: bool,
) -> pd.DataFrame:

    # ---------- serial path ----------
    if workers is None or workers <= 0 or not _HAS_JOBLIB:
        rows = []
        it = df.iterrows()
        if progress:
            it = _tqdm(it, total=len(df), desc="Rows")
        for _, r in it:
            rows.append(_safe_extract_row(
                r,
                groups=groups,
                lap_eigs_k=lap_eigs_k,
                netlsd_points=netlsd_points,
                netlsd_tmin=netlsd_tmin,
                netlsd_tmax=netlsd_tmax,
                ind4_exact_nmax=ind4_exact_nmax,
                ind4_max_samples=ind4_max_samples,
                ind5_sample_frac=ind5_sample_frac,
                distance_max_sources=distance_max_sources,
                distance_exact_if_leq=distance_exact_if_leq,
                seed=seed,
                intra_workers=intra_workers,
                betweenness_k=betweenness_k,
            ))
        df_rows = pd.DataFrame(rows)
        return _finalize_columns(df_rows)

    # ---------- parallel path ----------
    def _job(row: pd.Series):
        return _safe_extract_row(
            row,
            groups=groups,
            lap_eigs_k=lap_eigs_k,
            netlsd_points=netlsd_points,
            netlsd_tmin=netlsd_tmin,
            netlsd_tmax=netlsd_tmax,
            ind4_exact_nmax=ind4_exact_nmax,
            ind4_max_samples=ind4_max_samples,
            ind5_sample_frac=ind5_sample_frac,
            distance_max_sources=distance_max_sources,
            distance_exact_if_leq=distance_exact_if_leq,
            seed=seed,
            intra_workers=intra_workers,
            betweenness_k=betweenness_k,
        )

    iterator = (r for _, r in df.iterrows())

    if progress and _HAS_TQDM_JOBLIB:
        with tqdm_joblib(_tqdm(total=len(df), desc="Rows (parallel)")):
            results = Parallel(n_jobs=workers, backend=backend)(
                delayed(_job)(row) for row in iterator
            )
    else:
        results = Parallel(n_jobs=workers, backend=backend)(
            delayed(_job)(row) for row in iterator
        )

    df_rows = pd.DataFrame(results)
    return _finalize_columns(df_rows)


# ========================= IO helpers / manifest =========================
def infer_out_format(path: str, override: Optional[str]) -> str:
    if override:
        return override
    return "parquet" if path.lower().endswith(".parquet") else "csv"

def write_frame(df: pd.DataFrame, path: str, fmt: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)

def count_data_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)
    return max(total_lines - 1, 0)

def save_manifest(path: str, manifest: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

# ========================= CLI =========================
def main():
    ap = argparse.ArgumentParser(
        description="Graph feature extractor restricted to a specific feature set."
    )
    ap.add_argument("--input", required=True, help="Input CSV with EDGES or DEN_EDGES+NUM_EDGES.")
    ap.add_argument("--output", required=True, help="Output file path (csv or parquet). In batched mode, used for format inference.")
    ap.add_argument("--out-format", choices=["csv","parquet"], default=None,
                    help="Override output format (default inferred from --output extension).")

    # Batching + resume
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Rows per batch. If >0, stream input with chunksize and write each batch into --batch-dir.")
    ap.add_argument("--batch-dir", type=str, default=None,
                    help="Directory to write batched outputs. Defaults to '<output_basename>_batches'.")
    ap.add_argument("--from-batch", type=int, default=0,
                    help="Resume: 0-based batch index to start from (skips earlier batches if present).")

    # Groups & toggles
    ap.add_argument("--groups", type=str, default="all",
                    help="Comma-separated groups (or 'all'). "
                         f"Choices: {','.join(ALL_GROUPS)}")

    # Parallelism & UX
    ap.add_argument("--workers", type=int, default=0, help="Row-level parallel workers (joblib). 0=serial.")
    ap.add_argument("--backend", choices=["loky","threading"], default="loky", help="joblib backend.")
    ap.add_argument("--intra-workers", type=int, default=0, help="Intra-row workers for induced-4/5. 0=off.")
    ap.add_argument("--betweenness-k", type=int, default=0, help="If >0, use approximate betweenness with K sample nodes.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars.")

    # Spectral / NetLSD knobs
    ap.add_argument("--lap-eigs-k", type=int, default=10, help="How many Laplacian eigenvalues to keep (first K).")
    ap.add_argument("--netlsd-points", type=int, default=32, help="Number of log-spaced times for NetLSD.")
    ap.add_argument("--netlsd-tmin", type=float, default=0.01, help="Min time for NetLSD grid (log-spaced).")
    ap.add_argument("--netlsd-tmax", type=float, default=100.0, help="Max time for NetLSD grid (log-spaced).")

    # Induced motif sampling knobs
    ap.add_argument("--ind4-exact-nmax", type=int, default=50, help="Exact induced-4 if n <= this; else sample.")
    ap.add_argument("--ind4-max-samples", type=int, default=50000, help="Max sampled 4-sets when n > ind4-exact-nmax (0=auto).")
    ap.add_argument("--ind5-sample-frac", type=float, default=1.0, help="Fraction of 5-sets to sample for induced5 in (0,1].")

    # Distance sampling knobs
    ap.add_argument("--distance-max-sources", type=int, default=200, help="Sampled BFS sources for distance stats.")
    ap.add_argument("--distance-exact-if-leq", type=int, default=500, help="Use exact APSP if connected and n<=this.")

    # Misc
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--keep-coeff-first", action="store_true", help="Keep COEFFICIENTS column first if present.")



    ap.add_argument("--s3-uri", type=str, default=None,
                help="S3 destination (e.g. s3://physicsml/features/run-001/)")
    ap.add_argument("--stream-to-s3", action="store_true",
                    help="Stream output directly to S3 without saving locally")

    args = ap.parse_args()

    out_fmt = infer_out_format(args.output, args.out_format)
    groups = parse_groups(args.groups)
    progress = (not args.no_progress)

    # Non-batched mode
    if args.batch_size <= 0:
        df = pd.read_csv(args.input)

    

        feat_df = compute_batch_df(
            df,
            groups=groups,
            lap_eigs_k=args.lap_eigs_k,
            netlsd_points=args.netlsd_points,
            netlsd_tmin=args.netlsd_tmin,
            netlsd_tmax=args.netlsd_tmax,
            ind4_exact_nmax=args.ind4_exact_nmax,
            ind4_max_samples=args.ind4_max_samples,
            ind5_sample_frac=args.ind5_sample_frac,
            distance_max_sources=args.distance_max_sources,
            distance_exact_if_leq=args.distance_exact_if_leq,
            seed=args.seed,
            intra_workers=args.intra_workers,
            betweenness_k=args.betweenness_k,
            workers=args.workers,
            backend=args.backend,
            progress=progress,
        )
        if args.keep_coeff_first and "COEFFICIENTS" in feat_df.columns:
            cols = ["COEFFICIENTS"] + [c for c in feat_df.columns if c != "COEFFICIENTS"]
            feat_df = feat_df.reindex(columns=cols)
        #write_frame(feat_df, args.output, out_fmt)
        if args.s3_uri and args.stream_to_s3:
            fname = os.path.basename(args.output)
            write_frame_to_s3_stream(feat_df, args.s3_uri, fname, out_fmt)
            print(f"✓ Streamed to S3 → {args.s3_uri}/{fname}")
        else:
            write_frame(feat_df, args.output, out_fmt)
        print(f"✓ Feature table saved → {args.output}")

        # Manifest (single)
        manifest_path = f"{args.output}.manifest.json"
        manifest = {
            "mode": "single",
            "input": os.path.abspath(args.input),
            "output": os.path.abspath(args.output),
            "out_format": out_fmt,
            "rows_total": len(df),
            "settings": {
                "groups": groups,
                "workers": args.workers,
                "backend": args.backend,
                "intra_workers": args.intra_workers,
                "betweenness_k": args.betweenness_k,
                "lap_eigs_k": args.lap_eigs_k,
                "netlsd_points": args.netlsd_points,
                "netlsd_tmin": args.netlsd_tmin,
                "netlsd_tmax": args.netlsd_tmax,
                "ind4_exact_nmax": args.ind4_exact_nmax,
                "ind4_max_samples": args.ind4_max_samples,
                "ind5_sample_frac": args.ind5_sample_frac,
                "distance_max_sources": args.distance_max_sources,
                "distance_exact_if_leq": args.distance_exact_if_leq,
                "seed": args.seed,
                "keep_coeff_first": bool(args.keep_coeff_first),
            },
            "batches": [
                {
                    "index": 1,
                    "rows": {"start": 0, "end": len(df)-1, "count": len(df)},
                    "file": os.path.abspath(args.output),
                }
            ],
        }
        save_manifest(manifest_path, manifest)
        print(f"✓ Manifest saved → {manifest_path}")
        return

    # Batched mode
    batch_dir = args.batch_dir
    if not batch_dir:
        base = os.path.splitext(os.path.basename(args.output))[0]
        batch_dir = f"{base}_batches"
    os.makedirs(batch_dir, exist_ok=True)

    total_rows = count_data_rows(args.input)
    n_batches = (total_rows + args.batch_size - 1) // args.batch_size if args.batch_size > 0 else 0

    manifest_path = os.path.join(batch_dir, "manifest.json")
    manifest = {"mode":"batched","input":os.path.abspath(args.input),"batch_dir":os.path.abspath(batch_dir),
                "out_format": out_fmt,"rows_total": total_rows,"batch_size": args.batch_size,
                "settings":{
                    "groups": groups,"workers":args.workers,"backend":args.backend,
                    "intra_workers":args.intra_workers,"betweenness_k":args.betweenness_k,
                    "lap_eigs_k":args.lap_eigs_k,"netlsd_points":args.netlsd_points,
                    "netlsd_tmin":args.netlsd_tmin,"netlsd_tmax":args.netlsd_tmax,
                    "ind4_exact_nmax":args.ind4_exact_nmax,"ind4_max_samples":args.ind4_max_samples,
                    "ind5_sample_frac":args.ind5_sample_frac,
                    "distance_max_sources":args.distance_max_sources,
                    "distance_exact_if_leq":args.distance_exact_if_leq,
                    "seed":args.seed,"keep_coeff_first":bool(args.keep_coeff_first),
                },
                "batches":[]}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path,"r") as f:
                old = json.load(f)
            if isinstance(old, dict) and "batches" in old:
                manifest = old
        except Exception:
            pass

    start_row = args.from_batch * args.batch_size if args.from_batch > 0 else 0
    chunk_iter = pd.read_csv(args.input, chunksize=args.batch_size, skiprows=range(1, start_row+1)) \
                 if start_row > 0 else pd.read_csv(args.input, chunksize=args.batch_size)

    batch_index_start = args.from_batch if args.from_batch > 0 else 0
    batch_pbar = _tqdm(total=(n_batches - batch_index_start), desc="Batches") if progress else None
    batch_idx = batch_index_start
    total_rows_seen = start_row

    while True:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break

        start_idx = total_rows_seen
        end_idx = start_idx + len(chunk) - 1

        fname = f"feats_batch_{batch_idx:05d}_{start_idx:06d}-{end_idx:06d}.{('parquet' if out_fmt=='parquet' else 'csv')}"
        out_path = os.path.join(batch_dir, fname)

        if os.path.exists(out_path):
            if not any(b.get("file")==os.path.abspath(out_path) for b in manifest.get("batches", [])):
                manifest["batches"].append({
                    "index": int(batch_idx),
                    "rows": {"start": int(start_idx), "end": int(end_idx), "count": int(len(chunk))},
                    "file": os.path.abspath(out_path),
                })
                save_manifest(manifest_path, manifest)
            total_rows_seen += len(chunk)
            batch_idx += 1
            if batch_pbar: batch_pbar.update(1)
            continue

        feat_df = compute_batch_df(
            chunk,
            groups=groups,
            lap_eigs_k=args.lap_eigs_k,
            netlsd_points=args.netlsd_points,
            netlsd_tmin=args.netlsd_tmin,
            netlsd_tmax=args.netlsd_tmax,
            ind4_exact_nmax=args.ind4_exact_nmax,
            ind4_max_samples=args.ind4_max_samples,
            ind5_sample_frac=args.ind5_sample_frac,
            distance_max_sources=args.distance_max_sources,
            distance_exact_if_leq=args.distance_exact_if_leq,
            seed=args.seed,
            intra_workers=args.intra_workers,
            betweenness_k=args.betweenness_k,
            workers=args.workers,
            backend=args.backend,
            progress=progress,
        )

        if args.keep_coeff_first and "COEFFICIENTS" in feat_df.columns:
            cols = ["COEFFICIENTS"] + [c for c in feat_df.columns if c != "COEFFICIENTS"]
            feat_df = feat_df.reindex(columns=cols)

        #write_frame(feat_df, out_path, out_fmt)

        if args.s3_uri and args.stream_to_s3:
            fname = os.path.basename(out_path)
            write_frame_to_s3_stream(feat_df, args.s3_uri, fname, out_fmt)
            print(f"✓ Streamed batch → {args.s3_uri}/{fname}")
        else:
            write_frame(feat_df, out_path, out_fmt)

        manifest["batches"].append({
            "index": int(batch_idx),
            "rows": {"start": int(start_idx), "end": int(end_idx), "count": int(len(chunk))},
            "file": os.path.abspath(out_path),
        })
        save_manifest(manifest_path, manifest)

        total_rows_seen += len(chunk)
        batch_idx += 1
        if batch_pbar: batch_pbar.update(1)

    if batch_pbar: batch_pbar.close()
    print(f"✓ All batches written to: {batch_dir}")
    print(f"✓ Manifest saved → {manifest_path}")

if __name__ == "__main__":
    main()
