#!/usr/bin/env python3
# fgraph_features_cli.py
# ----------------------
# Single- or two-layer graph feature extractor with:
# - Graceful handling of missing/empty DEN/NUM layers
# - Feature groups (toggle via --groups)
# - Batched processing (--batch-size) with manifest JSON
# - Progress bars (tqdm), warning cleanup
#
# Examples:
#   python fgraph_features_cli.py --input edges.csv --output feats.csv
#   python fgraph_features_cli.py --input multilayer.csv --output feats.parquet --out-format parquet
#   python fgraph_features_cli.py --input huge.csv --output feats.csv --batch-size 10000 --batch-dir feats_batches
#
# Groups: --groups all (default) or comma list from:
#   basic,connectivity,centrality,core,robustness,cycles,
#   spectral_laplacian,spectral_adjacency,netlsd,planarity,symmetry,
#   community,motifs34,motifs5,induced4,induced5,tda

import os, argparse, ast, warnings, math, random, json
from collections import Counter, defaultdict
from itertools import combinations
from math import log2
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.stats import skew

# ---------- perf & warnings ----------
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
# Also silence scipy.stats ConstantInputWarning directly (pearsonr on constant arrays)
try:
    from scipy.stats import ConstantInputWarning as SciPyConstantInputWarning
    warnings.filterwarnings("ignore", category=SciPyConstantInputWarning)
except Exception:
    pass

# ---------- optional deps ----------
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

# TDA
try:
    from ripser import ripser
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False


# ========================= Feature groups =========================
ALL_GROUPS = [
    "basic","connectivity","centrality","core","robustness","cycles",
    "spectral_laplacian","spectral_adjacency","netlsd","planarity","symmetry",
    "community","motifs34","motifs5","induced4","induced5","tda",
]
DEFAULT_ENABLED_GROUPS = {k: True for k in ALL_GROUPS}


# ========================= Helpers =========================
def _tqdm(it, **kwargs):
    return tqdm(it, dynamic_ncols=True, mininterval=0.1, smoothing=0.1, leave=False, **kwargs)

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
    seen = set()
    faces = []
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
    """Return NaN (no warning) if degrees are constant or graph has no edges."""
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

# ---- induced 4-node ----
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

def _induced4_counts(G, *, max_samples=None, rng=None):
    counts = {"K1_3": 0, "P4": 0, "C4": 0, "TailedTriangle": 0, "Diamond": 0, "K4": 0}
    nodes = list(G.nodes()); n = len(nodes)
    if n < 4: return counts
    if max_samples is None:
        it = combinations(nodes, 4)
    else:
        rng = rng or random.Random(42)
        seen = set()
        total = math.comb(n, 4)
        k = min(max_samples, total)
        while len(seen) < k:
            S = tuple(sorted(rng.sample(nodes, 4)))
            if S not in seen: seen.add(S)
        it = list(seen)
    for S in it:
        H = G.subgraph(S)
        key = _classify_induced4(H)
        if key is not None:
            counts[key] += 1
    return counts


# ========================= Spectral helpers =========================
def _laplacian_eigs(G):
    L = nx.laplacian_matrix(G).todense()
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
    evals = np.linalg.eigvalsh(A)
    energy  = float(np.sum(np.abs(evals)))
    estrada = float(np.sum(np.exp(evals)))
    n = len(evals) if len(evals) > 0 else np.nan
    m2 = float(np.mean(evals**2)) if n == n and n > 0 else np.nan
    m3 = float(np.mean(evals**3)) if n == n and n > 0 else np.nan
    m4 = float(np.mean(evals**4)) if n == n and n > 0 else np.nan
    return energy, estrada, m2, m3, m4


# ========================= TDA helpers =========================
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
        total = float(np.sum(pers)); meanp = float(np.mean(pers)); maxp = float(np.max(pers))
        probs = pers / (total if total > 0 else 1.0)
        ent = -float(np.sum([p*np.log(p) for p in probs if p > 0]))
        return dict(count=float(finite.shape[0]), total=total, mean=meanp, max_=maxp, ent=ent,
                    mean_birth=float(np.mean(finite[:,0])), mean_death=float(np.mean(finite[:,1])))

    def _betti_at(dgm, t):
        if dgm.size == 0: return 0.0
        birth = dgm[:,0]; death = dgm[:,1]
        return float(np.sum((birth <= t) & (t < death)))

    H0_list, H1_list = [], []; betti0_q = [0.0,0.0,0.0]; betti1_q = [0.0,0.0,0.0]
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
        "TDA_Betti0_at_q25": betti0_q[0], "TDA_Betti1_at_q25": betti1_q[0],
        "TDA_Betti0_at_q50": betti0_q[1], "TDA_Betti1_at_q50": betti1_q[1],
        "TDA_Betti0_at_q75": betti0_q[2], "TDA_Betti1_at_q75": betti1_q[2],
    })
    return out


# ========================= 5-motifs =========================
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


# ========================= induced 5-node (WL) =========================
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
        names = [f"G5_{i:02d}" for i in range(len(reps))]
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

def _induced5_counts(G, *, sample_fraction: float = 1.0, random_state: int = 42):
    counts = {name: 0 for name in _G5_NAMES}
    if G.number_of_nodes() < 5 or not _G5_NAMES or not _HAS_WL:
        return counts, np.nan
    nodes = list(G.nodes()); comb_iter = combinations(nodes, 5)
    if sample_fraction < 1.0:
        rnd = random.Random(random_state)
        comb_iter = (c for c in comb_iter if rnd.random() < sample_fraction)
    connected_total = 0
    for S in comb_iter:
        H = G.subgraph(S)
        if not nx.is_connected(H): continue
        connected_total += 1
        try:
            h = wl_hash(H)
            name = _G5_HASH2NAME.get(h)
            if name is None:
                for rep, rep_name in zip(_G5_REPS, _G5_NAMES):
                    if nx.is_isomorphic(H, rep):
                        name = rep_name; _G5_HASH2NAME[h] = name; break
            if name is not None:
                counts[name] += 1
        except Exception:
            continue
    return counts, float(connected_total)


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
) -> Dict[str, Any]:

    rnd = random.Random(seed)
    enabled = lambda k: groups.get(k, False)
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
            "Basic_degree_skew": float(skew(degs))   if len(degs) > 2 else np.nan,
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

    # CONNECTIVITY + sampled distances
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
            bc = try_or_nan(nx.betweenness_centrality, G, normalized=True)
            cc = try_or_nan(nx.closeness_centrality,  G)
            try:
                ec = nx.eigenvector_centrality_numpy(G)
            except Exception:
                ec = np.nan
        def _stats(d):
            if isinstance(d, dict) and d:
                vals = list(d.values())
                return {"mean": np.mean(vals), "max": np.max(vals), "std": np.std(vals),
                        "skew": skew(vals) if len(vals) > 2 else np.nan}
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
            feats["Spectral_algebraic_connectivity"] = leigs[1]           if len(leigs) > 1 else np.nan
            feats["Spectral_spectral_gap"]           = leigs[1]-leigs[0]  if len(leigs) > 1 else np.nan
            feats["Spectral_laplacian_mean"]         = float(np.mean(leigs))
            feats["Spectral_laplacian_std"]          = float(np.std(leigs))
            feats["Spectral_laplacian_skew"]         = float(skew(leigs)) if len(leigs) > 2 else np.nan
            K = max(1, int(lap_eigs_k))
            pad = np.full(K, np.nan); pad[:min(K, len(leigs))] = leigs[:K]
            feats.update({f"Spectral_lap_eig_{i}": float(pad[i]) for i in range(K)})
            nonzero = leigs[leigs > 1e-12]
            feats["Kirchhoff_index"] = float(n * np.sum(1.0 / nonzero)) if len(nonzero) >= 1 else np.nan
        except Exception:
            feats.update({
                "Spectral_algebraic_connectivity": np.nan,
                "Spectral_spectral_gap":           np.nan,
                "Spectral_laplacian_mean":         np.nan,
                "Spectral_laplacian_std":          np.nan,
                "Spectral_laplacian_skew":         np.nan,
                **{f"Spectral_lap_eig_{i}": np.nan for i in range(max(1, int(lap_eigs_k)))},
                "Kirchhoff_index": np.nan,
            })
        feats.update(_laplacian_heat_traces(leigs, ts=(0.1, 1.0, 5.0)))
    else:
        feats.update({
            "Spectral_algebraic_connectivity": np.nan,
            "Spectral_spectral_gap":           np.nan,
            "Spectral_laplacian_mean":         np.nan,
            "Spectral_laplacian_std":          np.nan,
            "Spectral_laplacian_skew":         np.nan,
            **{f"Spectral_lap_eig_{i}": np.nan for i in range(max(1, int(lap_eigs_k)))},
            "Kirchhoff_index": np.nan,
            "Spectral_laplacian_heat_trace_t0.1": np.nan,
            "Spectral_laplacian_heat_trace_t1.0": np.nan,
            "Spectral_laplacian_heat_trace_t5.0": np.nan,
        })

    # NETLSD summaries
    if enabled("netlsd"):
        if leigs is not None and len(leigs) > 0 and not np.any(np.isnan(leigs)):
            ts = _lap_time_grid(netlsd_tmin, netlsd_tmax, netlsd_points)
            heat = np.exp(-np.outer(ts, leigs)).sum(axis=1)
            feats["NetLSD_mean"] = float(np.mean(heat))
            feats["NetLSD_std"]  = float(np.std(heat))
            feats["NetLSD_q10"]  = float(np.quantile(heat, 0.10))
            feats["NetLSD_q90"]  = float(np.quantile(heat, 0.90))
            feats["NetLSD_vector_json"] = json.dumps([float(x) for x in heat])
        else:
            feats["NetLSD_mean"] = np.nan
            feats["NetLSD_std"]  = np.nan
            feats["NetLSD_q10"]  = np.nan
            feats["NetLSD_q90"]  = np.nan
            feats["NetLSD_vector_json"] = json.dumps([])

    # PLANARITY
    if enabled("planarity"):
        try:
            planar, emb = nx.check_planarity(G)
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
            feats["Planarity_num_faces"]      = np.nan
            feats["Planarity_face_size_mean"] = np.nan
            feats["Planarity_face_size_max"]  = np.nan

    # SYMMETRY (heavy)
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
    if enabled("motifs34") or enabled("spectral_adjacency") or enabled("motifs5") or enabled("induced4") or enabled("induced5"):
        A = _adjacency_matrix(G)
    if enabled("motifs34"):
        tri = _triangle_count(G, A=A)
        feats["Motif_triangles"]   = tri
        feats["Motif_wedges"]      = _wedge_count(G, degs=[d for _, d in G.degree()], tri=tri)
        feats["Motif_4_cycles"]    = _four_cycle_count(G, A=A)
        feats["Motif_4_cliques"]   = _four_clique_count(G)
        m_mean, m_std = _edge_triangle_incidence_stats(G)
        feats["Motif_triangle_edge_incidence_mean"] = m_mean
        feats["Motif_triangle_edge_incidence_std"]  = m_std
        feats["Motif_square_clustering_proxy"]      = _square_clustering_proxy(G, A=A)
        emb_list = [len(set(G[u]) & set(G[v])) for u, v in G.edges()]
        feats["Motif_triangle_edge_incidence_median"] = _q(emb_list, 0.50)
        feats["Motif_triangle_edge_incidence_q90"]    = _q(emb_list, 0.90)
        feats["Motif_triangle_edge_frac_zero"]        = _frac(emb_list, lambda a: np.isclose(a, 0.0))
        feats["Motif_triangle_edge_frac_ge2"]         = _frac(emb_list, lambda a: a >= 2)

    # MOTIFS (5)
    if enabled("motifs5"):
        feats["Motif_5_cycles"]  = _five_cycle_count(G)
        feats["Motif_5_cliques"] = _five_clique_count(G)

    # INDUCED 4
    if enabled("induced4"):
        max_samples = None
        if n > ind4_exact_nmax:
            max_samples = ind4_max_samples if ind4_max_samples > 0 else 50000
        ind4 = _induced4_counts(G, max_samples=max_samples, rng=rnd)
        for k, v in ind4.items():
            feats[f"Motif_induced_{k}"] = float(v)
        total_4sets = safe_comb(n, 4) if n >= 4 else np.nan
        feats["Motif_induced_connected_per_4set"] = float(sum(ind4.values()) / total_4sets) if (total_4sets and total_4sets==total_4sets) else np.nan

    # INDUCED 5
    if enabled("induced5"):
        ind5_counts, connected5 = _induced5_counts(G, sample_fraction=ind5_sample_frac, random_state=seed)
        for name, val in ind5_counts.items():
            feats[f"Motif_induced5_{name}"] = float(val)
        total_5sets = safe_comb(n, 5) if n >= 5 else np.nan
        feats["Motif_induced_connected_per_5set"] = (connected5 / total_5sets) if (total_5sets and total_5sets==total_5sets and connected5==connected5) else np.nan

    # SPECTRAL (Adjacency/Energy)
    if enabled("spectral_adjacency"):
        A = _adjacency_matrix(G)
        energy, estrada, m2, m3, m4 = _adjacency_spectrum_features(A)
        feats["Adjacency_energy"]        = energy
        feats["Adjacency_estrada_index"] = estrada
        feats["Adjacency_moment_2"]      = m2
        feats["Adjacency_moment_3"]      = m3
        feats["Adjacency_moment_4"]      = m4

    # TDA
    if enabled("tda"):
        feats.update(_tda_vr_h_summary_from_spdm(G))

    # NORMALIZATIONS
    nf = float(n); mf = float(m)
    avgdeg = feats.get("Basic_avg_degree", np.nan)

    # Basic / connectivity norms
    if enabled("basic") or enabled("connectivity"):
        feats["Basic_avg_degree_norm"]       = feats.get("Basic_avg_degree", np.nan) / max(nf-1, 1) if nf==nf else np.nan
        feats["Basic_degree_entropy_norm"]   = feats.get("Basic_degree_entropy", np.nan) / np.log2(max(nf-1, 2)) if nf==nf else np.nan
        for col, denom in [("Connectivity_diameter", nf-1), ("Connectivity_radius", nf-1)]:
            v = feats.get(col, np.nan)
            feats[f"{col}_norm"] = v / max(denom, 1) if (v==v and nf==nf) else np.nan
        pairs = nf*(nf-1)/2.0 if nf==nf else np.nan
        wd = feats.get("Connectivity_wiener_index", np.nan)
        feats["Wiener_mean_distance"] = wd/pairs if (wd==wd and pairs and pairs==pairs) else np.nan

    # Motif norms
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


# ========================= Multi-layer wrappers =========================
def parse_edge_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple)):
        return [tuple(e) for e in x]
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return []
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [tuple(e) for e in obj]
        except Exception:
            return []
    return []

def build_graphs_from_row(row):
    """
    Returns one of:
      {"SINGLE": G}                        -- if EDGES provided
      {"DEN": G_DEN, "NUM": G_NUM, "TOTAL": G_DEN ∪ G_NUM}  -- if both non-empty
      {"DEN": G_DEN}                       -- if only DEN non-empty
      {"NUM": G_NUM}                       -- if only NUM non-empty
      {"SINGLE": empty_graph}              -- if all empty/missing
    """
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

    # All empty -> return empty graph so we still produce a row
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
            seed=seed,
        )
        if "COEFFICIENTS" in row: feats["COEFFICIENTS"] = row["COEFFICIENTS"]
        return feats

    # Layered path (DEN/NUM[/TOTAL]), but may be partial (only one layer)
    out: Dict[str, Any] = {}

    # Always compute for whichever layers exist
    for tag in ("DEN","NUM","TOTAL"):
        if tag in graphs:
            feats_tag = extract_features_single_graph(
                graphs[tag], groups,
                lap_eigs_k=lap_eigs_k, netlsd_points=netlsd_points,
                netlsd_tmin=netlsd_tmin, netlsd_tmax=netlsd_tmax,
                ind4_exact_nmax=ind4_exact_nmax, ind4_max_samples=ind4_max_samples,
                ind5_sample_frac=ind5_sample_frac,
                distance_max_sources=distance_max_sources,
                distance_exact_if_leq=distance_exact_if_leq,
                seed=seed,
            )
            out.update({f"{k}_{tag}": v for k, v in feats_tag.items()})

    # Cross-layer metrics only if BOTH DEN and NUM exist
    if ("DEN" in graphs) and ("NUM" in graphs):
        G_DEN, G_NUM = graphs["DEN"], graphs["NUM"]
        G_TOTAL = nx.compose(G_DEN, G_NUM)
        n = G_TOTAL.number_of_nodes()
        e_den, e_num = G_DEN.number_of_edges(), G_NUM.number_of_edges()
        union_edges = set(G_DEN.edges()) | set(G_NUM.edges())
        inter_edges = set(G_DEN.edges()) & set(G_NUM.edges())
        out["Edge_ratio_DEN_NUM"] = e_den / max(e_num, 1)
        out["Edge_overlap_count"] = len(inter_edges)
        out["Edge_jaccard"] = len(inter_edges) / max(len(union_edges), 1)
        out["Edge_overlap_frac_DEN"] = len(inter_edges) / max(e_den, 1)
        out["Edge_overlap_frac_NUM"] = len(inter_edges) / max(e_num, 1)
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

    if "COEFFICIENTS" in row: out["COEFFICIENTS"] = row["COEFFICIENTS"]
    return out


# ========================= Batch core =========================
def _safe_extract_row(row, **kwargs):
    try:
        return extract_features_row(row, **kwargs)
    except Exception as e:
        return {"__error__": str(e)}

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
    workers: int,
    backend: str,
    progress: bool,
) -> pd.DataFrame:

    if workers is None or workers <= 0 or not _HAS_JOBLIB:
        rows = []
        it = df.iterrows()
        it = _tqdm(it, total=len(df), desc="Rows") if progress else it
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
            ))
        return pd.DataFrame(rows)

    iterator = (
        _safe_extract_row(
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
        )
        for _, r in df.iterrows()
    )
    if progress and _HAS_TQDM_JOBLIB:
        with tqdm_joblib(_tqdm(total=len(df), desc="Rows (parallel)")):
            results = Parallel(n_jobs=workers, backend=backend)(
                delayed(lambda x: x)(x) for x in iterator
            )
    else:
        results = Parallel(n_jobs=workers, backend=backend)(
            list(delayed(lambda x: x)(x) for x in iterator)
        )
    return pd.DataFrame(results)


# ========================= CLI glue =========================
def parse_groups(arg_groups: str) -> Dict[str, bool]:
    if not arg_groups or arg_groups.lower() == "all":
        return DEFAULT_ENABLED_GROUPS.copy()
    base = {k: False for k in ALL_GROUPS}
    req = [s.strip().lower() for s in arg_groups.split(",") if s.strip()]
    for k in req:
        if k in base: base[k] = True
    return base

def infer_out_format(path: str, override: str | None) -> str:
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

def main():
    ap = argparse.ArgumentParser(
        description="Graph feature extractor (single- or two-layer) with full CLI knobs, batching, tqdm, manifest, and robust empty-layer handling."
    )
    ap.add_argument("--input", required=True, help="Input CSV with EDGES or DEN_EDGES+NUM_EDGES.")
    ap.add_argument("--output", required=True, help="Output file path (csv or parquet). In batched mode, used for format inference.")
    ap.add_argument("--out-format", choices=["csv","parquet"], default=None,
                    help="Override output format (default inferred from --output extension).")

    # Batching
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Rows per batch. If >0, stream input with chunksize and write each batch into --batch-dir.")
    ap.add_argument("--batch-dir", type=str, default=None,
                    help="Directory to write batched outputs. Defaults to '<output_basename>_batches'.")

    # Groups & toggles
    ap.add_argument("--groups", type=str, default="all",
                    help="Comma-separated groups (or 'all'). "
                         f"Choices: {','.join(ALL_GROUPS)}")

    # Parallelism & UX
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (joblib). 0=serial.")
    ap.add_argument("--backend", choices=["loky","threading"], default="loky", help="joblib backend.")
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
            workers=args.workers,
            backend=args.backend,
            progress=progress,
        )
        if args.keep_coeff_first and "COEFFICIENTS" in feat_df.columns:
            cols = ["COEFFICIENTS"] + [c for c in feat_df.columns if c != "COEFFICIENTS"]
            feat_df = feat_df.reindex(columns=cols)
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

    total_rows = count_data_rows(args.input)  # for a proper progress bar
    n_batches = (total_rows + args.batch_size - 1) // args.batch_size if args.batch_size > 0 else 0

    manifest = {
        "mode": "batched",
        "input": os.path.abspath(args.input),
        "batch_dir": os.path.abspath(batch_dir),
        "out_format": out_fmt,
        "rows_total": total_rows,
        "batch_size": args.batch_size,
        "settings": {
            "groups": groups,
            "workers": args.workers,
            "backend": args.backend,
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
        },
        "batches": [],
    }

    total_rows_seen = 0
    batch_idx = 1
    chunk_iter = pd.read_csv(args.input, chunksize=args.batch_size)

    batch_pbar = _tqdm(range(n_batches), total=n_batches, desc="Batches") if progress else None
    for _ in (range(n_batches) if progress else [None]*n_batches):
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break
        start_idx = total_rows_seen
        end_idx = total_rows_seen + len(chunk) - 1

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
            workers=args.workers,
            backend=args.backend,
            progress=progress,
        )

        if args.keep_coeff_first and "COEFFICIENTS" in feat_df.columns:
            cols = ["COEFFICIENTS"] + [c for c in feat_df.columns if c != "COEFFICIENTS"]
            feat_df = feat_df.reindex(columns=cols)

        fname = f"feats_batch_{batch_idx:05d}_{start_idx:06d}-{end_idx:06d}.{('parquet' if out_fmt=='parquet' else 'csv')}"
        out_path = os.path.join(batch_dir, fname)
        write_frame(feat_df, out_path, out_fmt)

        manifest["batches"].append({
            "index": batch_idx,
            "rows": {"start": int(start_idx), "end": int(end_idx), "count": int(len(chunk))},
            "file": os.path.abspath(out_path),
        })

        total_rows_seen += len(chunk)
        batch_idx += 1
        if batch_pbar: batch_pbar.update(1)

    if batch_pbar: batch_pbar.close()

    # Save manifest
    manifest_path = os.path.join(batch_dir, "manifest.json")
    save_manifest(manifest_path, manifest)
    print(f"✓ All batches written to: {batch_dir}")
    print(f"✓ Manifest saved → {manifest_path}")

# ----------------------------
if __name__ == "__main__":
    main()
