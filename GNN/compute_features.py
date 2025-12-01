# -----------------------------------------------------------
# 0. HPC-SAFE PARALLEL SETTINGS (MUST COME FIRST!)
# -----------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["GEOMINE_OMP_THREADS"] = "1"

# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------
import numpy as np
import pandas as pd
import networkx as nx
import ast
from pathlib import Path
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import psutil
from joblib import Parallel, delayed


# ===========================================================
# 1.LOAD DATA
# ===========================================================


def load_graph_edges(file_ext='7', base_dir=None):
    base_dir = Path(base_dir or '../Graph_Edge_Data')
    npz_path = base_dir / f'den_graph_data_{file_ext}.npz'
    csv_path = base_dir / f'den_graph_data_{file_ext}.csv'

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        return data['edges'].tolist(), data['coefficients'].tolist()

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        edges_list = [ast.literal_eval(e) for e in df['EDGES']]
        coeffs = df['COEFFICIENTS'].tolist()
        return edges_list, coeffs

    raise FileNotFoundError(f"No graph data found for loop {file_ext}")


def edges_to_networkx(edges):
    nodes = sorted(set(u for e in edges for u in e))
    mapping = {n: i for i,n in enumerate(nodes)}
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for u,v in edges:
        G.add_edge(mapping[u], mapping[v])
    return G, len(nodes)

# ===========================================================
# 2. HPC-AWARE AUTOTUNING (PER FEATURE)
# ===========================================================

import psutil

def get_node_specs():
    """Return (n_cpus, mem_gb) for the current node."""
    n_cpus = psutil.cpu_count(logical=True) or 1
    mem_gb = psutil.virtual_memory().available / 1e9  # FREE memory
    return n_cpus, mem_gb


# -----------------------------------------------------------
# Rules for each feature class
# -----------------------------------------------------------

AUTOTUNE_RULES = {
    "graphlet_4": {
        "base_n_jobs": 16,
        "base_chunk": 10000,
        "scale_with_cpu": False,
        "scale_with_mem": False,
    },
    "graphlet_3": {
        "base_n_jobs": 16,
        "base_chunk": 20000,
        "scale_with_cpu": True,   # allow 1–2 jobs depending on CPU
        "scale_with_mem": False,
    },
    "eigen": {
        "base_n_jobs": 16,
        "base_chunk": 20000,
        "scale_with_cpu": True,
        "scale_with_mem": True,   # eigen benefits from larger chunks if RAM is available
    },
    "degree": {
        "base_n_jobs": 24,
        "base_chunk": 50000,
        "scale_with_cpu": True,
        "scale_with_mem": True,
    },
    "clustering": {
        "base_n_jobs": 24,
        "base_chunk": 50000,
        "scale_with_cpu": True,
        "scale_with_mem": True,
    },
    # Add more specific handlers here
}


def detect_feature_class(feature):
    """Map a feature name to its autotune class."""
    f = feature.lower()
    for key in AUTOTUNE_RULES:
        if key in f:
            return key
    return "degree"  # default for light features


# -----------------------------------------------------------
# Autotuning for a *single* feature
# -----------------------------------------------------------

def autotune_for_feature(feature):
    n_cpus, mem_gb = get_node_specs()
    fclass = detect_feature_class(feature)
    rules = AUTOTUNE_RULES[fclass]

    # Start from base values
    n_jobs = min(rules["base_n_jobs"], n_cpus)

    # Scale n_jobs with CPU count?
    if rules["scale_with_cpu"]:
        # e.g. with 48 CPUs, eigen may use 4–8
        n_jobs = min(n_jobs + (n_cpus // 32), n_cpus)

    chunk = rules["base_chunk"]

    # Scale chunk size with memory?
    if rules["scale_with_mem"]:
        # Safe memory scaling: grows slowly with RAM
        if mem_gb > 60:               # fat nodes
            chunk *= 2
        elif mem_gb > 120:            # rare nodes
            chunk *= 4

    # Safety: never use more jobs than CPUs
    n_jobs = min(n_jobs, n_cpus)

    return chunk, n_jobs


# -----------------------------------------------------------
# Autotuning for a list of features
# -----------------------------------------------------------

def autotune_all_features(feature_list):
    """Return a dict: feature -> (chunk_size, n_jobs)."""
    settings = {}
    for feat in feature_list:
        chunk, jobs = autotune_for_feature(feat)
        settings[feat] = {"chunk_size": chunk, "n_jobs": jobs}
    return settings



# ===========================================================
# 3. FEATURE FUNCTIONS
# ===========================================================

# ---------- Degree ----------
def compute_degree_features(batch):
    out = []
    for edges in batch:
        G, n = edges_to_networkx(edges)
        out.append([G.degree(i) for i in range(n)])
    return out

# ---------- Adjacency ----------
def adjacency_column_features(batch):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        A = nx.adjacency_matrix(G, nodelist=range(n)).toarray()
        out.append([A[:,i] for i in range(n)])
    return out

# ---------- Identity ----------
def identity_column_features(batch):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        I = np.eye(n)
        out.append([I[:,i] for i in range(n)])
    return out

# ---------- Betweenness, clustering, closeness, pagerank ----------
def compute_betweenness_features(batch):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        b = nx.betweenness_centrality(G)
        out.append([b[i] for i in range(n)])
    return out

def compute_clustering_features(batch):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        c = nx.clustering(G)
        out.append([c[i] for i in range(n)])
    return out

def compute_closeness_features(batch):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        c = nx.closeness_centrality(G)
        out.append([c[i] for i in range(n)])
    return out

def compute_pagerank_features(batch):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        p = nx.pagerank(G)
        out.append([p[i] for i in range(n)])
    return out

# ---------- Graphlets via GEOMINE ----------
def compute_graphlet_features(batch, k=4, sizev=1, sizee=2, connect=True):
    try:
        import GEOMINE
    except Exception:
        raise RuntimeError("GEOMINE missing. Install via pip.")

    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        A = nx.adjacency_matrix(G, nodelist=range(n)).toarray()
        node_feats=[]
        for ref in range(n):
            vec = GEOMINE.Count(A, ref, k, sizev, sizee, connect)
            node_feats.append(np.asarray(vec, float))
        out.append(node_feats)
    return out

def graphlet_3(batch): return compute_graphlet_features(batch, k=3)
def graphlet_4(batch): return compute_graphlet_features(batch, k=4)
#
# def graphlet_5(batch): return compute_graphlet_features(batch, k=5)

# ---------- Eigenvectors ----------
def compute_extreme_eigenvectors(batch, k_low=3, k_high=3):
    out=[]
    for edges in batch:
        G,n = edges_to_networkx(edges)
        L = nx.laplacian_matrix(G).astype(float)

        try:
            vals_low, vecs_low = eigsh(L, k=min(k_low+1,n-1), which='SM')
            vals_high, vecs_high = eigsh(L, k=min(k_high,n-1), which='LM')
        except:
            Ld = L.toarray()
            vals, vecs = eigh(Ld)
            idx = np.argsort(vals)
            vals_low, vecs_low = vals[idx], vecs[:,idx]
            vals_high, vecs_high = vals[idx], vecs[:,idx]

        nz = np.where(vals_low > 1e-10)[0][:k_low]
        low_vecs = vecs_low[:,nz] if len(nz)>0 else np.zeros((n,k_low))
        high_vecs = vecs_high[:, -k_high:] if vecs_high.shape[1]>=k_high else np.zeros((n,k_high))

        out.append(np.hstack([low_vecs, high_vecs]))
    return out

def create_eigen_feature_function(i, kind='low', k_low=3, k_high=3):
    def f(batch):
        all_feats = compute_extreme_eigenvectors(batch, k_low, k_high)
        if kind=='low': return [x[:,i-1].tolist() for x in all_feats]
        else: return [x[:,k_low + i - 1].tolist() for x in all_feats]
    return f

# register eigen functions
low_eigenvector_functions={f'low_eigen_{i}': create_eigen_feature_function(i,'low') for i in range(1,4)}
eigenvector_functions={f'eigen_{i}': create_eigen_feature_function(i,'high') for i in range(1,4)}

# ===========================================================
# 4. FEATURE REGISTRY
# ===========================================================

FEATURE_FUNCTIONS = {
    **eigenvector_functions,
    **low_eigenvector_functions,
    'degree': compute_degree_features,
    'adjacency_columns': adjacency_column_features,
    'identity_columns': identity_column_features,
    'betweenness': compute_betweenness_features,
    'clustering': compute_clustering_features,
    'closeness': compute_closeness_features,
    'pagerank': compute_pagerank_features,
    'graphlet_3': graphlet_3,
    'graphlet_4': graphlet_4,
    #'graphlet_5': graphlet_5
}

# ===========================================================
# 5. PAD FEATURES FOR CONSISTENT SHAPES
# ===========================================================

def pad_features(features_list, max_nodes):
    padded=[]
    for feat in features_list:
        A = np.asarray(feat)
        n = A.shape[0]
        if n < max_nodes:
            pad = (max_nodes-n,) + A.shape[1:]
            A = np.vstack([A, np.zeros(pad)])
        else:
            A = A[:max_nodes]
        padded.append(A)
    return np.array(padded)

# ===========================================================
# 6. MAIN COMPUTATION ENGINE
# ===========================================================

def compute_or_load_features(file_ext, selected_features, base_dir, chunk_size, n_jobs):
    """
    Compute or load selected node features for denominator graphs.

    Priority order for each feature:
        1. feat.npy exists                → load it
        2. partial files exist            → attempt merge → save feat.npy
        3. stored inside .npz archive     → extract → save feat.npy
        4. otherwise compute now

    HPC behavior:
        - If --range_start / --range_end is used → slice_mode (compute only part & save part files)
        - If --hpc_mode but no slice args → merge-only mode (only merge partials, NEVER compute)
    """

    base_dir = Path(base_dir)
    feat_dir = base_dir / f"features_loop_{file_ext}"
    feat_dir.mkdir(exist_ok=True)

    npz_path = feat_dir / f"features_loop_{file_ext}.npz"

    # ------------------------------------------------------
    # Detect HPC slice mode
    # ------------------------------------------------------
    slice_mode = (args.range_start is not None) or (args.range_end is not None)
    merge_only = (args.hpc_mode and not slice_mode)

    # Load full edges (needed for compute + merge logic)
    full_edges_list, _ = load_graph_edges(file_ext, base_dir)
    total_graphs = len(full_edges_list)
    max_nodes = max(len(set(u for e in edges for u in e)) for edges in full_edges_list)

    # Determine slice range (HPC or full)
    i0 = args.range_start if args.range_start is not None else 0
    i1 = args.range_end if args.range_end is not None else total_graphs

    edges_list = full_edges_list[i0:i1]
    print(f"🔎 Processing graphs {i0} → {i1} (chunk={chunk_size}, jobs={n_jobs}, slice_mode={slice_mode})")

    # =====================================================
    # STEP 1 — list all local .npy and partial files
    # =====================================================
    full_npys = {f.stem for f in feat_dir.glob("*.npy") if "_part_" not in f.name}

    partial_map = {}  # feat → [Path(part files)]
    for f in feat_dir.glob("*_part_*.npy"):
        stem = f.stem
        feat = stem.split("_part_")[0]
        partial_map.setdefault(feat, []).append(f)

    # =====================================================
    # STEP 2 — load NPZ archive only after checking local files
    # =====================================================
    existing_npz = {}
    if npz_path.exists():
        print(f"📦 Found archive {npz_path.name}")
        with np.load(npz_path, allow_pickle=True) as z:
            existing_npz = {k: z[k] for k in z.files}
        print(f"📦 Archive contains: {list(existing_npz.keys())}")

    # =====================================================
    # STEP 3 — helper to detect feature status
    # =====================================================
    def get_status(feat):
        if feat in full_npys:
            return "full"
        if feat in partial_map:
            return "partial"
        if feat in existing_npz:
            return "npz"
        return "missing"

    # =====================================================
    # STEP 4 — merge partial function
    # =====================================================
    def try_merge_partials(feat):
        parts = partial_map.get(feat, [])
        if not parts:
            return None

        # Parse start/end
        ranges = []
        for p in parts:
            toks = p.stem.split("_")
            try:
                start = int(toks[-2])
                end = int(toks[-1])
                ranges.append((start, end, p))
            except:
                continue

        if not ranges:
            return None

        # Sort by start
        ranges.sort(key=lambda x: x[0])

        # Must cover exactly [0, total_graphs]
        if ranges[0][0] != 0:
            return None
        for (_, end1, _), (start2, _, _) in zip(ranges, ranges[1:]):
            if end1 != start2:
                return None
        if ranges[-1][1] != total_graphs:
            return None

        print(f"🔧 Merging {len(ranges)} partials for {feat}...")

        blocks = [np.load(p) for _, _, p in ranges]
        merged = np.concatenate(blocks, axis=0)

        np.save(feat_dir / f"{feat}.npy", merged)
        print(f"✨ Saved full merged feature → {feat}.npy")
        return merged

    # =====================================================
    # STEP 5 — compute feature helper
    # =====================================================
    def compute_feature(feat):
        func = FEATURE_FUNCTIONS[feat]
        all_out = []

        for idx in tqdm(range(0, len(edges_list), chunk_size), desc=f"Computing {feat}"):
            chunk = edges_list[idx : idx + chunk_size]

            if n_jobs > 1:
                subs = [chunk[j:j+100] for j in range(0, len(chunk), 100)]
                results = Parallel(n_jobs=n_jobs)(
                    delayed(func)(s) for s in subs
                )
                out = [x for r in results for x in r]
            else:
                out = func(chunk)

            all_out.extend(out)

        arr = pad_features(all_out, max_nodes)

        # partial save (HPC) or full save (local)
        if slice_mode:
            out_file = feat_dir / f"{feat}_part_{i0}_{i1}.npy"
        else:
            out_file = feat_dir / f"{feat}.npy"

        np.save(out_file, arr)
        print(f"💾 Saved {feat} → {out_file}")

        return arr

    # =====================================================
    # STEP 6 — handle each feature in priority order
    # =====================================================
    for feat in selected_features:
        status = get_status(feat)
        print(f"\n➡️ Feature '{feat}' has status: {status}")

        # 1. full exists
        if status == "full":
            print(f"✔ Using full file {feat}.npy")
            existing_npz[feat] = np.load(feat_dir / f"{feat}.npy")
            continue

        # 2. partial exists → try merge
        if status == "partial":
            merged = try_merge_partials(feat)
            if merged is not None:
                existing_npz[feat] = merged
                continue
            print(f"⚠️ Partials incomplete → must compute {feat}")

            if merge_only:
                print(f"❌ merge-only mode: refusing to recompute {feat}")
                continue

            arr = compute_feature(feat)
            if not slice_mode:
                existing_npz[feat] = arr
            continue

        # 3. in archive only
        if status == "npz":
            print(f"📎 Extracting {feat} from archive → writing .npy")
            np.save(feat_dir / f"{feat}.npy", existing_npz[feat])
            continue

        # 4. completely missing
        print(f"🧮 Computing missing {feat}")
        if merge_only:
            print(f"❌ merge-only mode: refusing to compute {feat}")
            continue

        arr = compute_feature(feat)
        if not slice_mode:
            existing_npz[feat] = arr

    # =====================================================
    # STEP 7 — save NPZ only in local mode
    # =====================================================
    if not slice_mode:
        print("💾 Updating NPZ...")
        np.savez_compressed(npz_path, **existing_npz)
        print(f"🎉 NPZ saved → {npz_path}")
    else:
        print("🧱 HPC slice mode → skipping NPZ write.")



            
            




# ===========================================================
#  CLI ENTRYPOINT (supports HPC slicing + merge-only mode)
# ===========================================================

if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser()

    # ---------------- CLI arguments ----------------
    parser.add_argument('--config', type=str, help='YAML config file')
    parser.add_argument('--file_ext', type=str, help='Loop order override')
    parser.add_argument('--feature', type=str, help='Compute only this feature')

    parser.add_argument('--chunk_size', type=int, help='Override chunk size')
    parser.add_argument('--n_jobs', type=int, help='Override job parallelism')

    # HPC slicing
    parser.add_argument('--range_start', type=int, help='Start index for HPC array')
    parser.add_argument('--range_end', type=int, help='End index for HPC array')
    parser.add_argument('--hpc-mode', action='store_true',
                        help='Enable merge/slice HPC behaviour')

    # Disable autotuning
    parser.add_argument('--no-autotune', action='store_true',
                        help='Disable autotuning of chunk & n_jobs')

    args = parser.parse_args()

    # ---------------- Debug print ----------------
    print("DEBUG ARGS:")
    for k, v in vars(args).items():
        print(f"  {k:12} = {v}")
    print()

    # ---------------- Load YAML config ----------------
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # ---------------- Extract data paths ----------------
    base_dir = Path(config.get('data', {}).get('base_dir', '../Graph_Edge_Data'))
    file_ext = args.file_ext if args.file_ext else config.get('data', {}).get('file_ext')

    # ---------------- Determine feature list ----------------
    if args.feature:
        # explicit override → compute only this one
        features_to_run = [args.feature]

    else:
        cfg_feats = config.get('features', {})

        # Option 1: compute_all = true
        if cfg_feats.get('compute_all', False):
            features_to_run = list(FEATURE_FUNCTIONS.keys())

        # Option 2: YAML contains a features_list
        elif 'features_list' in cfg_feats:
            features_to_run = cfg_feats['features_list']

        # Option 3: fallback → compute everything
        else:
            print("⚠️ No 'features_list' found — computing all known features.")
            features_to_run = list(FEATURE_FUNCTIONS.keys())

    print(f"\n### Features to compute: {features_to_run}")

    # ---------------- Loop over features ----------------
    for feature_name in features_to_run:
        print(f"\n🚀 {feature_name}")

        # ---------------- Determine chunk & jobs ----------------
        # User overrides take priority
        if args.chunk_size or args.n_jobs:
            chunk_size = (
                args.chunk_size
                if args.chunk_size is not None
                else config['features'].get('chunk_size', 1000)
            )
            n_jobs = (
                args.n_jobs
                if args.n_jobs is not None
                else config['features'].get('n_jobs', 1)
            )

        # Autotune using the new API
        elif not args.no_autotune:
            chunk_size, n_jobs = autotune_for_feature(feature_name)

        # Fallback if autotune disabled
        else:
            chunk_size = config['features'].get('chunk_size', 1000)
            n_jobs = config['features'].get('n_jobs', 1)

        print(f"   → chunk={chunk_size}, n_jobs={n_jobs}")

        # ---------------- Timing wrapper ----------------
        t0 = time.perf_counter()

        compute_or_load_features(
            file_ext=file_ext,
            selected_features=[feature_name],
            base_dir=base_dir,
            chunk_size=chunk_size,
            n_jobs=n_jobs
        )

        dt = time.perf_counter() - t0
        print(f"⏱ Finished {feature_name} in {dt:.2f} seconds")
